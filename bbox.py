import numpy as np
detections=[[(-25, -12), (178, 50), 0.0794057622551918, 1.0], [(-1, -7), (208, 57), 0.05563075840473175, 1.0], [(205, -19), (397, 60), 0.09535861760377884, 1.0], [(251, -28), (397, 69), 0.1595219522714615, 1.0], [(4, 0), (166, 82), 0.010843622498214245, 0.0], [(4, 0), (166, 82), 0.26842761039733887, 1.0], [(23, -12), (210, 95), 0.010040754452347755, 0.0], [(23, -12), (210, 95), 0.21227556467056274, 1.0], [(224, -6), (383, 94), 0.012419145554304123, 0.0], [(224, -6), (383, 94), 0.30395635962486267, 1.0], [(254, -9), (399, 96), 0.01873016729950905, 0.0], [(254, -9), (399, 96), 0.45904341340065, 1.0], [(-6, 30), (196, 124), 0.018894236534833908, 1.0], [(12, 39), (214, 113), 0.025693494826555252, 1.0], [(214, 16), (404, 135), 0.01656969077885151, 1.0], [(255, 42), (410, 117), 0.07368412613868713, 1.0], [(-25, 63), (191, 168), 0.0647905170917511, 1.0], [(10, 70), (205, 155), 0.044433288276195526, 1.0], [(222, 60), (395, 154), 0.03619512543082237, 1.0], [(267, 70), (398, 156), 0.01052078790962696, 0.0], [(267, 70), (398, 156), 0.27721884846687317, 1.0], [(-21, 97), (195, 178), 0.04358862712979317, 1.0]]
for top_left_corner,bottom_right_corner, conf, class_id in detections:
    #if confidence_treshold < 0.1: continue
    top_left_list=list(top_left_corner)
    bottom_right_list=list(bottom_right_corner)
    if top_left_list[0] < 0:
        top_left_list[0]=0
    if top_left_list[1] < 0:
        top_left_list[1]=0
    if bottom_right_list[0] < 0:
        bottom_right_list[0]=0
    if bottom_right_list[1] < 0:
        bottom_right_list[1]=0    
    centerx=round((top_left_list[0]+top_left_list[1])/2)
    centery=round((bottom_right_list[0]+bottom_right_list[1])/2)
    height=round(bottom_right_list[1]-top_left_list[1])
    width=round(bottom_right_list[0]-top_left_list[0])
    print( f"bbox: {top_left_corner},{bottom_right_corner}, score: {conf}, class_id: {class_id}")
    print("normalized bounding_boxes")    
    print( f"bbox: {top_left_list},{bottom_right_list}, score: {conf}, class_id: {class_id}")
    print("converted coordinates")
    print("centx ",centerx," centy",centery," height",height," width",width)
