from datetime import datetime, timezone
from pydantic import BaseModel

class Test(BaseModel):
    dt: datetime

# Create a UTC datetime
utc_time = datetime.now(timezone.utc)
t = Test(dt=utc_time)

print("Original datetime:", utc_time)
print("mode=python:", t.model_dump(mode='python')['dt'])
print("mode=json:", t.model_dump(mode='json')['dt'])
print("model_dump_json:", t.model_dump_json())
