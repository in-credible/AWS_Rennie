#!/usr/bin/env python
# w-adebayo
from constructs import Construct
from cdktf import App, TerraformStack

from imports.aws import AwsProvider, SnsTopic, CloudwatchMetricAlarm

class MyStack(TerraformStack):
    def __init__(self, scope: Construct, ns: str):
        super().__init__(scope, ns)

        # define resources here
        AwsProvider(self, 'Aws', region='us-east-1')

        # SNS Topic
        BlogTopic = SnsTopic(self, 'Topic', display_name='w-ade-blog-cdktf')

        # CloudWatch Alarm
        CloudwatchMetricAlarm(self, 'w-adeBlogAlarm',
                                actions_enabled     = True,
                                alarm_actions       = [BlogTopic.arn],
                                alarm_name          = 'w-ade-blog-cdktf',
                                comparison_operator = 'GreaterThanOrEqualToThreshold',
                                evaluation_periods  = 1,
                                metric_name         = 'VpcEventCount',
                                namespace           = 'CloudTrailMetrics',
                                period              = 300,
                                statistic           = 'Sum',
                                threshold           = 1,
                                treat_missing_data  = 'notBreaching'
                              )

app = App()
MyStack(app, "tf-rocks")

app.synth()
