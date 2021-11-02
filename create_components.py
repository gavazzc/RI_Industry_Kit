import genericpath
import re
import sys
import argparse
import os
import shutil
import boto3
import yaml
import zipfile
import os.path
from os import path
import time

list_processed_component = []
general_d = {}

def zip(src, dst):
    zf = zipfile.ZipFile("%s.zip" % (dst), "w", zipfile.ZIP_DEFLATED)
    abs_src = os.path.abspath(src)
    for dirname, subdirs, files in os.walk(src):
        for filename in files:
            absname = os.path.abspath(os.path.join(dirname, filename))
            arcname = absname[len(abs_src) + 1:]
            print('zipping %s as %s' % (os.path.join(dirname, filename),
                                        arcname))
            zf.write(absname, arcname)
    zf.close()

def create_components(name):
    if (name not in list_processed_component):
        for component_recipe in os.listdir(build_recipes_path):
            recipe_file_path = os.path.join(build_recipes_path, component_recipe)
            with open(recipe_file_path) as f:
                try:
                    response = gg_client.create_component_version(inlineRecipe=f.read())
                    print(response)
                except Exception as e:
                    print("Failed to create the component using the recipe at {}.\nException: {}".format(recipe_file_path, e))
                    exit(1)

def upload_artifacts(file_name, object_name=None):
    if object_name is None:
        object_name = file_name
    try:
        print("Uploading artifacts to the bucket {} with key {}.".format(bucket, object_name))
        result = s3_client.upload_file(file_name, bucket, object_name)
        print(result)
    except Exception as e:
        print("Failed to upload the artifacts to the bucket {} with key {}.\nException: {}".format(bucket, object_name, e))
        exit(1)
    print("Successfully uploaded the artifacts to the bucket {} with key {}.".format(bucket, object_name))

def create_artifacts(name):
    if (name not in list_processed_component):
        for component in os.listdir(artifacts_path):
            component_path = os.path.join(artifacts_path, component)       

            for version in os.listdir(component_path):
                if version.startswith("."):
                    continue
                version_path = os.path.join(component_path, version)
                component_latest_version = get_latest_version(component, version)
                build = os.path.join(build_artifacts_path, component, component_latest_version)
                os.makedirs(build, mode=0o777, exist_ok = True)
                zip(version_path+"/", build+"/"+name)
                upload_artifacts(build+"/"+name+".zip", "artifacts/"+name+".zip")
                return "artifacts/"+name+".zip"


def create_recipes_with_artifacts(res, name):

    if (name not in list_processed_component):
        os.makedirs(build_recipes_path, mode=0o777, exist_ok=True)
        for component_recipe in os.listdir(recipes_path):
            component = component_recipe.split("-")
            c_name = component[0]
            c_version = component[1].split(".yaml")[0]
            latest_version = get_latest_version(c_name, c_version)
            print(latest_version)
            with open(os.path.join(recipes_path, component_recipe)) as f:
                recipe = f.read()
            recipe = recipe.replace("$BUCKETNAME$", bucket)
            recipe = recipe.replace("$COMPONENT_VERSION$", latest_version)
            recipe = recipe.replace("$PATHZIP$", res)

            recipe = yaml.load(recipe)
            recipe["ComponentVersion"]=latest_version
            d[c_name] = {'componentVersion': recipe["ComponentVersion"]}
            general_d[c_name] = {'componentRecipes': component_recipe, 'componentVersion': recipe["ComponentVersion"]}

            recipe_file_name = "{}-{}.json".format(c_name, latest_version)
            with open(os.path.join(build_recipes_path, recipe_file_name), "w") as f:
                f.write(yaml.dump(recipe))        

    else:
        for component_recipe in os.listdir(recipes_path):
            component = component_recipe.split("-")
            c_name = component[0]
            d[c_name] = {'componentVersion': general_d[c_name]['componentVersion']}


def create_recipes_without_artifacts(name):
    if (name not in list_processed_component):
        os.makedirs(build_recipes_path, mode=0o777, exist_ok=True)
        for component_recipe in os.listdir(recipes_path):
            component = component_recipe.split("-")
            c_name = component[0]    
            c_version = component[1].split(".yaml")[0]

            latest_version = get_latest_version(c_name, c_version)
            print(latest_version)
            with open(os.path.join(recipes_path, component_recipe)) as f:
                recipe = f.read()
            recipe = recipe.replace("$COMPONENT_VERSION$", latest_version)

            recipe = yaml.load(recipe)
            recipe["ComponentVersion"]=latest_version
            d[c_name] = {'componentVersion': recipe["ComponentVersion"]}
            general_d[c_name] = {'componentRecipes': component_recipe, 'componentVersion': recipe["ComponentVersion"]}

            recipe_file_name = "{}-{}.json".format(c_name, latest_version)
            with open(os.path.join(build_recipes_path, recipe_file_name), "w") as f:
                f.write(yaml.dump(recipe))
    else:
        
        for component_recipe in os.listdir(recipes_path):
            component = component_recipe.split("-")
            c_name = component[0] 
            d[c_name] = {'componentVersion': general_d[c_name]['componentVersion']}


def get_account_number():
    try:
        response = sts_client.get_caller_identity()
        if response is not None:
            account = response["Account"]
            return account
    except Exception as e:
        print("Cannot get the account id from the credentials.\nException: {}".format(e))
        exit(1)

def get_latest_version(c_name, c_version):
    try:
        account = get_account_number()
        print("Fetching the component {} from the account: {}, region: {}".format(c_name, account, region))
        response = gg_client.list_component_versions(
            arn="arn:aws:greengrass:{}:{}:components:{}".format(region, account, c_name),
        )
        if response is not None:
            print(response)
            component_versions = response["componentVersions"]
            if component_versions:
                versions = component_versions[0]["componentVersion"]
                split = versions.split("-")[0].split(".")
                major = split[0]
                minor = split[1]
                patch = split[2]
                return "{}.{}.{}".format(major, minor, str(int(patch) + 1))
            else:
                return c_version
    except Exception as e:
        print("Error getting the latest version of the component.\nException: {}".format(e))

def create_distribution(d, g_name):
    d['aws.greengrass.Cli'] = {'componentVersion': '2.3.0'}
    #d['aws.greengrass.SecureTunneling'] = {'componentVersion': '1.0.3'}
    d['aws.greengrass.Nucleus'] = {'componentVersion': '2.3.0'}
    response = gg_client.create_deployment(
            targetArn=gname_prefix+g_name,
            deploymentName='InstancesGroupAWS',
            components=d,            
            deploymentPolicies={
                'failureHandlingPolicy': 'DO_NOTHING',
                'componentUpdatePolicy': {
                    'timeoutInSeconds': 123,
                    'action': 'SKIP_NOTIFY_COMPONENTS'
                },
                'configurationValidationPolicy': {
                    'timeoutInSeconds': 123
                }
            }
        )
    print("\n\n")
    print(d)

cpath = "Components/"
gname_prefix = "arn:aws:iot:eu-central-1:**************:thinggroup/"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--cmps_path",
    default=cpath,
    help="You must specify the folder where the components are located",
)

args = parser.parse_args()
rootdir = args.cmps_path

if(not path.exists(rootdir)):
    print("The path is not valid.")
    exit(1)

session = boto3.Session(
    region_name="eu-central-1",
    aws_access_key_id='',
    aws_secret_access_key=''
)

gg_client = session.client("greengrassv2")
sts_client = session.client("sts")
s3_client = session.client("s3")
region = "eu-central-1"
bucket = "ggv2-example-component-artifacts"

groups = os.listdir(rootdir)
for group in groups:
    d = {}
    g = os.listdir(rootdir+"/"+group)
    print("\n\nPROCESSING GROUP:", group)
    for component in g:
        dir_path = os.path.dirname(os.path.realpath(__file__))+"/"+rootdir+"/"+group+"/"+component
        artifacts_path = os.path.join(dir_path, "artifacts")
        recipes_path = os.path.join(dir_path, "recipes")

        build_dir_path = os.path.join(dir_path, "build")
        build_artifacts_path = os.path.join(build_dir_path, "artifacts")
        build_recipes_path = os.path.join(build_dir_path, "recipes")

        shutil.rmtree(build_dir_path, ignore_errors=True, onerror=None)

        if(os.path.exists(artifacts_path)):
            res = create_artifacts(component)
            create_recipes_with_artifacts(res, component)
            create_components(component)
        else:
            create_recipes_without_artifacts(component)
            create_components(component)

        list_processed_component.append(component)

    create_distribution(d, group)
