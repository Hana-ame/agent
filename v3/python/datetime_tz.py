from datetime import datetime, timezone
import zoneinfo

# Python 3.9+ 推荐使用 zoneinfo
try:
    tz = zoneinfo.ZoneInfo("Asia/Shanghai")
    now = datetime.now(tz)
    print("Shanghai time:", now)
    
    # UTC 转换
    utc_now = now.astimezone(timezone.utc)
    print("UTC time:", utc_now)
except ImportError:
    # 旧版本 fallback
    import pytz
    tz = pytz.timezone("Asia/Shanghai")
    now = datetime.now(tz)
    print("Shanghai time (pytz):", now)
