#!/usr/bin/env node
import * as cdk from '@aws-cdk/core';

import { RiIkitStack } from '../lib/ri_ikit-stack';

const app = new cdk.App();
new RiIkitStack(app, 'RiIkitStack', {
      env: { account: process.env.CDK_DEFAULT_ACCOUNT, region: process.env.CDK_DEFAULT_REGION }
});