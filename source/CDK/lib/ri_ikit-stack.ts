import { Stack, StackProps, Construct, Duration, RemovalPolicy } from '@aws-cdk/core';
import { readFileSync } from 'fs';
import * as ec2 from "@aws-cdk/aws-ec2";
import * as iam from "@aws-cdk/aws-iam";
import * as s3 from "@aws-cdk/aws-s3";
import * as s3d from "@aws-cdk/aws-s3-deployment";
import * as elb from "@aws-cdk/aws-elasticloadbalancingv2";
import * as cloudfront from "@aws-cdk/aws-cloudfront";
import * as origins from "@aws-cdk/aws-cloudfront-origins";
import * as ec2t from "@aws-cdk/aws-elasticloadbalancingv2-targets";
import * as acm from "@aws-cdk/aws-certificatemanager";
import * as acmpca from "@aws-cdk/aws-acmpca";

//Its not possibile to automate the creation of the private certificate authority, please create one and substitute this variable
let acmPca = "arn:aws:acm-pca:eu-central-1:351032530776:certificate-authority/437cc382-3c00-4c56-85c1-fb262e79fd2d";

export class RiIkitStack extends Stack {
  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);

    /* Creation of the VPC to host the AWS resources */
    let vpc: ec2.Vpc = this.provisionVPC();
    
    /* Creation of the role for both ASGs */
    let sg: ec2.ISecurityGroup = this.getEC2SG(vpc);
    
    /* Creation of the role for the EC2s to access required services */
    let role: iam.Role = this.getEC2Role();
    
    this.getIoTRole();

    let ec2=this.provisionEC2Node(vpc,sg,role);
    
    let bucket=this.prepareS3Content();
    
    this.createLoadBalancer(ec2,vpc,sg);
    
    this.createCloudFormation(bucket);

  }

  private createCloudFormation(bucket: s3.Bucket){

    const distribution = new cloudfront.Distribution(this, 'rikit-s3-static', {
      defaultBehavior: {
        origin: new origins.S3Origin(bucket) // This class automatically creates an Origin Access Identity
      },
    });
    
  }  
  
  private createLoadBalancer(target: ec2.Instance, vpc: ec2.Vpc, sg: ec2.ISecurityGroup){
    
    //Create security group for the ELB
    let elbSG: ec2.SecurityGroup = new ec2.SecurityGroup(this, "sg-elb-rikit", {
      securityGroupName: "elb-rikit",
      vpc: vpc
    });
    //Enable me to access from internet
    elbSG.addIngressRule(ec2.Peer.ipv4('54.239.6.189/31'), ec2.Port.tcp(443));
    //Enable ELB to reach out to the EC2 on the 443
    elbSG.addEgressRule(sg, ec2.Port.tcp(443));
    //Enable EC2 to get reached by the ELB
    sg.addIngressRule(elbSG, ec2.Port.tcp(443));
    
    //Create private certificate for exposing HTTPS
    let certificate=new acm.PrivateCertificate(this, 'PrivateCertificate', {
      domainName: 'rikit.com',
      certificateAuthority: acmpca.CertificateAuthority.fromCertificateAuthorityArn(this, 'CA', acmPca),
    });
    
    const lb = new elb.ApplicationLoadBalancer(this, 'rikit', {
      vpc,
      internetFacing: true,
      securityGroup: elbSG
    });
    
    const listener = lb.addListener('Listener', {
      port: 443,
      open: true,
      certificates: [certificate]
    });

    const instanceTarget = new ec2t.InstanceTarget(target, 443);

    listener.addTargets('ApplicationFleet', {
      port: 443,
      targets: [instanceTarget]
    });

    return lb;
  }
  
  private prepareS3Content(){

    const ggv2Bucket = new s3.Bucket(this, 'ri-kit-ggv2', {publicReadAccess: false, blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,  removalPolicy: RemovalPolicy.DESTROY, autoDeleteObjects: true,});

    new s3d.BucketDeployment(this, 'bucketGGv2Content', {
      sources: [s3d.Source.asset('../Components'),s3d.Source.asset('../Images')],
      destinationBucket: ggv2Bucket,
      destinationKeyPrefix: 'RI_Industry_Kit'
    });
    
    const webBucket = new s3.Bucket(this, 'ri-kit-web', {publicReadAccess: false, blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,  removalPolicy: RemovalPolicy.DESTROY, autoDeleteObjects: true,});

    new s3d.BucketDeployment(this, 'bucketWebContent', {
      sources: [s3d.Source.asset('../Static')],
      destinationBucket: webBucket,
      destinationKeyPrefix: 'Web'
    });
    
    return webBucket;
  }
  
  private getEC2SG(vpc: ec2.Vpc) {

    let primarySG: ec2.SecurityGroup = new ec2.SecurityGroup(this, "sg-ec2-rikit", {
      securityGroupName: "ec2-rikit",
      vpc: vpc
    });

    return primarySG;
  }

  private getEC2Role() {

    // Setting the list of the policies necessary to enable the EC2 to work
    let managedPolicies: iam.IManagedPolicy[] = [
      iam.ManagedPolicy.fromAwsManagedPolicyName("AmazonSSMManagedInstanceCore"),
      iam.ManagedPolicy.fromAwsManagedPolicyName("AmazonEC2ReadOnlyAccess"),
      iam.ManagedPolicy.fromAwsManagedPolicyName("AWSGreengrassFullAccess"),
      iam.ManagedPolicy.fromAwsManagedPolicyName("AmazonS3FullAccess"),
      iam.ManagedPolicy.fromAwsManagedPolicyName("AWSIoTFullAccess"),
      iam.ManagedPolicy.fromAwsManagedPolicyName("IAMReadOnlyAccess")
    ];

    let customPolicy = new iam.PolicyDocument({
      statements: [new iam.PolicyStatement({actions: ["iam:PassRole",
        "elasticloadbalancing:DescribeLoadBalancers",
        "acm:ListCertificates",
        "acm:ExportCertificate",
        "acm:GetCertificate"], resources: ['*']})]
    });

    // Role to be assumed by EC2 instances
    let assumedBy: iam.IPrincipal = new iam.ServicePrincipal("ec2.amazonaws.com");

    // Setting the role for the EC2 that are going to spawn
    let ec2Role: iam.Role = new iam.Role(this, "sf-cluster-ec2-role", {
      managedPolicies: managedPolicies,
      inlinePolicies: {ri_ggv2_ec2: customPolicy},
      assumedBy: assumedBy
    });

    return ec2Role;
  }
  
  private getIoTRole() {

    // Setting the list of the policies necessary to enable the EC2 to work
    let managedPolicies: iam.IManagedPolicy[] = [
      iam.ManagedPolicy.fromAwsManagedPolicyName("AmazonS3FullAccess"),
    ];
    let customPolicy = new iam.PolicyDocument({
      statements: [new iam.PolicyStatement({actions: [
        "iot:AttachPrincipalPolicy",
        "iot:CreateKeysAndCertificate",
        "iot:CreatePolicy",
        "iot:DeleteCertificate",
        "iot:DeletePolicy",
        "iot:DetachPrincipalPolicy",
        "iot:GetPolicy",
        "iot:ListPolicyPrincipals",
        "iot:UpdateCertificate",
        "iot:DescribeCertificate",
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents",
        "logs:DescribeLogStreams"],
      resources: ['*']})]
    });

    // Setting the role for the EC2 that are going to spawn
    let ec2Role: iam.Role = new iam.Role(this, "ri-iot-ggv2-role", {
      managedPolicies: managedPolicies,
      inlinePolicies: {ri_ggv2_iot: customPolicy},
      assumedBy: new iam.ServicePrincipal("credentials.iot.amazonaws.com")
    });

    return ec2Role;
  }

  private provisionVPC() {

    // Public network group
    let publicSubnet: ec2.SubnetConfiguration = {
      cidrMask: 24,
      name: "sf_public",
      reserved: false,
      subnetType: ec2.SubnetType.PUBLIC
    };

    // Private network group
    let privateSubnet: ec2.SubnetConfiguration = {
      cidrMask: 24,
      name: "sf_private",
      reserved: false,
      subnetType: ec2.SubnetType.PRIVATE_WITH_NAT
    };

    // Configuring networks to add to the VPC
    let subnetConfigs: ec2.SubnetConfiguration[] = [publicSubnet, privateSubnet];

    /* Create VPC with the configured networks */
    let vpc: ec2.Vpc = new ec2.Vpc(this, "sf_vpc", {
      cidr: "10.0.0.0/20",
      maxAzs: 3,
      subnetConfiguration: subnetConfigs
    });

    return vpc;
  }
  
    /* Subnet selection for the cluster, this statement selects all the private subnets of all AZs in the region */
  private getPrimarySubnets() {

    let privateSubnets: ec2.SubnetSelection = { subnetType: ec2.SubnetType.PRIVATE_WITH_NAT };
    return privateSubnets;
  }
  
  private provisionEC2Node(vpc: ec2.Vpc, sg: ec2.ISecurityGroup, role: iam.Role) {

    // Fetch user data for primary asg
    let primaryFirstSFUserData: ec2.UserData = ec2.UserData.custom(this.getBootScript());

    return new ec2.Instance(this, "ri_kit_ggv2", {
      vpc: vpc,
      vpcSubnets: this.getPrimarySubnets(),
      instanceName: "ri_kit_ggv2",
      role: role,
      userData: primaryFirstSFUserData,
      machineImage: this.getPrimaryAMI(),
      securityGroup: sg,
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.BURSTABLE2, ec2.InstanceSize.MEDIUM)
    });
  }
    
  /* Read the local file at run time */
  private getBootScript() {

    let result: string = readFileSync('./resources/RIBootScript.txt', 'utf-8');
    /* The script is missing the <powershell> header so we can add dynamic variabiles and specific code for each type of instance generated */
    return result;
  }

  private getPrimaryAMI() {

    // Setup properties of the primary cluster AMIs
    let primaryServiceFabricAMI: ec2.IMachineImage = new ec2.LookupMachineImage({
      name: "amzn2-ami-hvm-2.0.20211001.1-x86_64-gp2",
      windows: false
    });

    return primaryServiceFabricAMI;
  }
}