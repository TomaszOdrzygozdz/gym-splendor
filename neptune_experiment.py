import math

import neptune

from neptune_settings import NEPTUNE_API_TOKEN, NEPTUNE_PROJECT_NAME

neptune.init(project_qualified_name=NEPTUNE_PROJECT_NAME, api_token=NEPTUNE_API_TOKEN)
neptune.create_experiment(name='Test1')

for i in range(1000):
    neptune.send_metric('metric_1', x=math.sin(i/100))
    neptune.se

neptune.stop()
