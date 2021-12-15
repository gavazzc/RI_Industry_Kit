import math
import time
from typing import Any

import exifread
import flirimageextractor
import geopandas as gpd
import numpy as np
import onnxruntime
import torch
import torchvision  # noqa: F401
import utm
from PIL import Image
from shapely.geometry import Polygon

session = None
def detect_onnx(
    image_path: str,
    model_path: str,
    anchors: list,
    num_classes: int,
    official: bool = True,
    get_metadata: bool = True,
):

    """
    Return detections for image or each image in folder.

    Args:
        official: Boolean value.
        image_path: Path of the image.
        model_path: Path of the model.
        anchors: Matrix of the predefined anchors.
        num_classes: Number of classes of the model.
    """
    global session
    if session is None:
        session = onnxruntime.InferenceSession(model_path)
    # print("The model expects input shape: ", session.get_inputs()[0].shape)
    batch_size = session.get_inputs()[0].shape[0]
    img_size_h = session.get_inputs()[0].shape[2]
    img_size_w = session.get_inputs()[0].shape[3]

    # input
    image_src = Image.open(image_path)
    width_orig, height_orig = image_src.size
    resized = letterbox_image(image_src, (img_size_w, img_size_h))

    img_in = np.transpose(resized, (2, 0, 1)).astype(np.float32)  # HWC -> CHW
    img_in = np.expand_dims(img_in, axis=0)
    img_in /= 255.0

    # inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_in})

    batch_detections: Any = []
    if official and len(outputs) == 4:
        # model.model[-1].export = False ---> outputs[0] (1, xxxx, 85)
        batch_detections = torch.from_numpy(np.array(outputs[0]))
        batch_detections = non_max_suppression(
            batch_detections, conf_thres=0.4, iou_thres=0.5, agnostic=False
        )
    else:

        boxs = []
        a = torch.tensor(anchors).float().view(3, -1, 2)
        anchor_grid = a.clone().view(3, 1, -1, 1, 1, 2)
        if len(outputs) == 4:
            outputs = [outputs[1], outputs[2], outputs[3]]
        for index, out in enumerate(outputs):
            out = torch.from_numpy(out)
            # batch = out.shape[1]
            feature_w = out.shape[2]
            feature_h = out.shape[3]

            # Feature map corresponds to the original image zoom factor
            stride_w = int(img_size_w / feature_w)
            stride_h = int(img_size_h / feature_h)

            grid_x, grid_y = np.meshgrid(np.arange(feature_w), np.arange(feature_h))

            # cx, cy, w, h
            pred_boxes = torch.FloatTensor(out[..., :4].shape)
            pred_boxes[..., 0] = (
                torch.sigmoid(out[..., 0]) * 2.0 - 0.5 + grid_x
            ) * stride_w  # cx
            pred_boxes[..., 1] = (
                torch.sigmoid(out[..., 1]) * 2.0 - 0.5 + grid_y
            ) * stride_h  # cy
            pred_boxes[..., 2:4] = (
                torch.sigmoid(out[..., 2:4]) * 2
            ) ** 2 * anchor_grid[
                index
            ]  # wh

            conf = torch.sigmoid(out[..., 4])
            pred_cls = torch.sigmoid(out[..., 5:])

            output = torch.cat(
                (
                    pred_boxes.view(batch_size, -1, 4),
                    conf.view(batch_size, -1, 1),
                    pred_cls.view(batch_size, -1, num_classes),
                ),
                -1,
            )
            boxs.append(output)

        outputx = torch.cat(boxs, 1)
        # NMS
        batch_detections = w_non_max_suppression(
            outputx, num_classes, conf_thres=0.4, nms_thres=0.3
        )
    if batch_detections[0]==None:
        result = {"detection": []}
    else:   
        labels = batch_detections[0][..., -1]
        boxs = batch_detections[0][..., :4]
        confs = batch_detections[0][..., 4].numpy()
        boxs[:, :] = scale_coords((640, 640), boxs, (height_orig, width_orig)).round()
        boxs = [list(x) for x in boxs.numpy()]
        for i, box in enumerate(boxs):
            box.extend([confs[i], int(labels[i])])
        result = {"detection": boxs}
    if get_metadata:
        result["metadata"] = get_metadata_exif(image_path)
    return result


def get_metadata_exif(image_path):
    flir = flirimageextractor.FlirImageExtractor()
    metadata = flir.get_metadata(image_path)
    with open(image_path, "rb") as f:
        tags = exifread.process_file(f)
        metadata["GPS GPSLatitude"] = tags["GPS GPSLatitude"]
        metadata["GPS GPSLongitude"] = tags["GPS GPSLongitude"]
        metadata["GPS GPSLatitudeRef"] = tags["GPS GPSLatitudeRef"]
        metadata["GPS GPSLongitudeRef"] = tags["GPS GPSLongitudeRef"]
    return metadata


def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new("RGB", size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


def non_max_suppression(
    prediction, conf_thres=0.1, iou_thres=0.6, merge=False, classes=None, agnostic=False
):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    max_wh = 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torch.ops.torchvision.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(
                    1, keepdim=True
                )  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy
            except Exception:
                # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(x, i, x.shape, i.shape)
                pass

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]
    # where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def w_non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):

    box_corner = torch.FloatTensor(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]

        if not image_pred.size(0):
            continue

        class_conf, class_pred = torch.max(
            image_pred[:, 5 : 5 + num_classes], 1, keepdim=True
        )

        detections = torch.cat(
            (image_pred[:, :5], class_conf.float(), class_pred.float()), 1
        )

        unique_labels = detections[:, -1].cpu().unique()

        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()

        for c in unique_labels:
            detections_class = detections[detections[:, -1] == c]
            _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]
            max_detections = []
            while detections_class.size(0):
                max_detections.append(detections_class[0].unsqueeze(0))
                if len(detections_class) == 1:
                    break
                ious = w_bbox_iou(max_detections[-1], detections_class[1:])
                detections_class = detections_class[1:][ious < nms_thres]
            max_detections = torch.cat(max_detections).data
            output[image_i] = (
                max_detections
                if output[image_i] is None
                else torch.cat((output[image_i], max_detections))
            )

    return output


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (
        (
            torch.min(box1[:, None, 2:], box2[:, 2:])
            - torch.max(box1[:, None, :2], box2[:, :2])
        )
        .clamp(0)
        .prod(2)
    )
    return inter / (area1[:, None] + area2 - inter)


def w_bbox_iou(box1, box2, x1y1x2y2=True):

    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )

    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def utmToLatLng(zone, easting, northing, northernHemisphere=True):
    if not northernHemisphere:
        northing = 10000000 - northing
    a = 6378137
    e = 0.081819191
    e1sq = 0.006739497
    k0 = 0.9996
    arc = northing / k0
    mu = arc / (
        a
        * (
            1
            - math.pow(e, 2) / 4.0
            - 3 * math.pow(e, 4) / 64.0
            - 5 * math.pow(e, 6) / 256.0
        )
    )
    ei = (1 - math.pow((1 - e * e), (1 / 2.0))) / (1 + math.pow((1 - e * e), (1 / 2.0)))
    ca = 3 * ei / 2 - 27 * math.pow(ei, 3) / 32.0
    cb = 21 * math.pow(ei, 2) / 16 - 55 * math.pow(ei, 4) / 32
    cc = 151 * math.pow(ei, 3) / 96
    cd = 1097 * math.pow(ei, 4) / 512
    phi1 = (
        mu
        + ca * math.sin(2 * mu)
        + cb * math.sin(4 * mu)
        + cc * math.sin(6 * mu)
        + cd * math.sin(8 * mu)
    )
    n0 = a / math.pow((1 - math.pow((e * math.sin(phi1)), 2)), (1 / 2.0))
    r0 = a * (1 - e * e) / math.pow((1 - math.pow((e * math.sin(phi1)), 2)), (3 / 2.0))
    fact1 = n0 * math.tan(phi1) / r0
    _a1 = 500000 - easting
    dd0 = _a1 / (n0 * k0)
    fact2 = dd0 * dd0 / 2
    t0 = math.pow(math.tan(phi1), 2)
    Q0 = e1sq * math.pow(math.cos(phi1), 2)
    fact3 = (5 + 3 * t0 + 10 * Q0 - 4 * Q0 * Q0 - 9 * e1sq) * math.pow(dd0, 4) / 24
    fact4 = (
        (61 + 90 * t0 + 298 * Q0 + 45 * t0 * t0 - 252 * e1sq - 3 * Q0 * Q0)
        * math.pow(dd0, 6)
        / 720
    )
    lof1 = _a1 / (n0 * k0)
    lof2 = (1 + 2 * t0 + Q0) * math.pow(dd0, 3) / 6.0
    lof3 = (
        (5 - 2 * Q0 + 28 * t0 - 3 * math.pow(Q0, 2) + 8 * e1sq + 24 * math.pow(t0, 2))
        * math.pow(dd0, 5)
        / 120
    )
    _a2 = (lof1 - lof2 + lof3) / math.cos(phi1)
    _a3 = _a2 * 180 / math.pi
    latitude = 180 * (phi1 - fact1 * (fact2 + fact3 + fact4)) / math.pi
    if not northernHemisphere:
        latitude = -latitude
    longitude = ((zone > 0) and (6 * zone - 183.0) or 3.0) - _a3
    return (latitude, longitude)


def degress(tag):
    d = float(tag.values[0].num) / float(tag.values[0].den)
    m = float(tag.values[1].num) / float(tag.values[1].den)
    s = float(tag.values[2].num) / float(tag.values[2].den)
    return d + (m / 60.0) + (s / 3600.0)


def world_file_generation(metadata, sensor_width=0.01):
    pw = metadata["ImageWidth"]
    ph = metadata["ImageHeight"]
    fl = float(metadata["FocalLength"][:-3]) / 1000
    sw = sh = 2 * fl * math.tan(45 * 3.1415 / 360)  # sensor_width
    alt = metadata["RelativeAltitude"]
    alpha = metadata["GimbalYawDegree"]
    lat = degress(metadata["GPS GPSLatitude"])
    lon = degress(metadata["GPS GPSLongitude"])
    lat = -lat if metadata["GPS GPSLatitudeRef"].values[0] != "N" else lat
    lon = -lon if metadata["GPS GPSLongitudeRef"].values[0] != "E" else lon
    utm_EN = utm.from_latlon(lat, lon)
    EAST = utm_EN[0]
    NORTH = utm_EN[1]
    ZONE = utm_EN[2]
    if utm_EN[3] in ["N", "O", "P", "Q", "R", "S", "T", "U", "V", "Z"]:
        EMISPHERE = True
    else:
        EMISPHERE = False
    a1 = fl / alt  # scale, dimensionless
    a2 = 1 / a1  # scale factor
    b1 = sw / pw  # meters per pixel on sensor's width
    c1 = sh / ph  # meters per pixel on sensor's height
    if c1 > b1:
        b1 = c1
    elif b1 > c1:
        c1 = b1
    b2 = ((pw / 2) * b1) * a2  # offset from central pixel in width direction in meters
    c2 = ((ph / 2) * c1) * a2  # offset from central pixel in height direction in meters
    czx = b1 * a2  # cellsize in width direction in meters
    czy = c1 * a2  # cellsize in width direction in meters
    if EMISPHERE:
        E1 = EAST + (b2)  # upper left pixel's X-coordinate in meters
        N1 = NORTH + (c2)  # upper left pixel's y-coordinate in meters
    else:
        E1 = EAST - (b2)  # upper left pixel's X-coordinate in meters
        N1 = NORTH + (c2)  # upper left pixel's y-coordinate in meters
    alpha = -alpha
    A = np.cos(np.deg2rad(alpha)) * czx
    D = np.sin(np.deg2rad(alpha)) * czy
    B = np.sin(np.deg2rad(alpha)) * czx
    E = -np.cos(np.deg2rad(alpha)) * czy
    E = -A
    return A, D, B, E, E1, N1, ZONE, EMISPHERE


def polygon_generation(image_name, anomaly, A, D, B, E, E1, N1, ZONE, EMISPHERE):
    x0 = A * (anomaly["left"]) + B * (anomaly["top"]) + E1
    y0 = D * (anomaly["left"]) + E * (anomaly["top"]) + N1
    x1 = A * (anomaly["right"]) + B * (anomaly["bottom"]) + E1
    y1 = D * (anomaly["right"]) + E * (anomaly["bottom"]) + N1
    return [
        1,
        image_name,
        anomaly["anomaly"],
        x0,
        x1,
        y0,
        y1,
        Polygon(
            [
                tuple(utmToLatLng(ZONE, x0, y0, northernHemisphere=EMISPHERE))[::-1],
                tuple(utmToLatLng(ZONE, x1, y0, northernHemisphere=EMISPHERE))[::-1],
                tuple(utmToLatLng(ZONE, x1, y1, northernHemisphere=EMISPHERE))[::-1],
                tuple(utmToLatLng(ZONE, x0, y1, northernHemisphere=EMISPHERE))[::-1],
            ]
        ),
    ]


def intersect_polygons(polygons_results, ZONE, EMISPHERE):
    for i in range(len(polygons_results)):
        if polygons_results[i][0] == 1:
            for j in range(i + 1, len(polygons_results)):
                if polygons_results[j][0] == 1:
                    if polygons_results[i][2] == polygons_results[j][2]:
                        if polygons_results[i][7].intersects(polygons_results[j][7]):
                            polygons_results[j][0] = 0
                            polygons_results[i][3] = min(
                                polygons_results[i][3], polygons_results[j][3]
                            )
                            polygons_results[i][4] = max(
                                polygons_results[i][4], polygons_results[j][4]
                            )
                            polygons_results[i][5] = max(
                                polygons_results[i][5], polygons_results[j][5]
                            )
                            polygons_results[i][6] = min(
                                polygons_results[i][6], polygons_results[j][6]
                            )
                            polygons_results[i][7] = Polygon(
                                [
                                    tuple(
                                        utmToLatLng(
                                            ZONE,
                                            polygons_results[i][3],
                                            polygons_results[i][5],
                                            northernHemisphere=EMISPHERE,
                                        )
                                    )[::-1],
                                    tuple(
                                        utmToLatLng(
                                            ZONE,
                                            polygons_results[i][4],
                                            polygons_results[i][5],
                                            northernHemisphere=EMISPHERE,
                                        )
                                    )[::-1],
                                    tuple(
                                        utmToLatLng(
                                            ZONE,
                                            polygons_results[i][4],
                                            polygons_results[i][6],
                                            northernHemisphere=EMISPHERE,
                                        )
                                    )[::-1],
                                    tuple(
                                        utmToLatLng(
                                            ZONE,
                                            polygons_results[i][3],
                                            polygons_results[i][6],
                                            northernHemisphere=EMISPHERE,
                                        )
                                    )[::-1],
                                ]
                            )
    pol_gdf = gpd.GeoDataFrame(
        {
            "anomaly": [x[2] for x in polygons_results],
            "flag": [x[0] for x in polygons_results],
        },
        geometry=[x[-1] for x in polygons_results],
    )
    subset = pol_gdf[pol_gdf["flag"] == 1].drop(["flag"], axis=1)
    return subset


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
            img1_shape[0] - img0_shape[0] * gain
        ) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2
