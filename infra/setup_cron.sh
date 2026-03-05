#!/bin/bash
CRON_LINE="30 0 * * 1-5 cd /Users/panda/Projects/japan-momentum && /usr/bin/python3 run_daily.py >> logs/daily.log 2>&1"
(crontab -l 2>/dev/null | grep -v japan-momentum; echo "$CRON_LINE") | crontab -
echo "cron設定完了:"
crontab -l | grep japan-momentum
