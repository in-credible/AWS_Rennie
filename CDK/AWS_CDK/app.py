#!/usr/bin/env python3
# Wade Bayo

from aws_cdk import core

from automation_rocks.automation_rocks_stack import (
        AutomationRocksStack,
        nova,
        ohio,
        )


app = core.App()
AutomationRocksStack(app, "automation-rocks-nova", env=nova)

AutomationRocksStack(app, "automation-rocks-ohio", env=ohio)

app.synth()
