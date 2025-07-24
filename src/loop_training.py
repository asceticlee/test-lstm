import subprocess
from datetime import datetime, timedelta

def str_to_date(s):
    return datetime.strptime(s, "%Y%m%d")

def date_to_str(d):
    return d.strftime("%Y%m%d")

# Initial dates
dates = [
    "20240929", "20241123",  # trainFrom, trainTo
    "20241124", "20241130",  # validationFrom, validationTo
    "20241201", "20241207"   # testFrom, testTo
]

# End condition
end_date = datetime(2025, 7, 22)

while True:
    # Run the script
    cmd = [
        "python", "stockprice_lstm_tensorflow_regression.py",
        *dates
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # Shift all dates by 7 days
    new_dates = [date_to_str(str_to_date(d) + timedelta(days=7)) for d in dates]

    # Check if testTo (last date) exceeds end_date
    if str_to_date(new_dates[-1]) > end_date:
        print("Reached end date. Stopping.")
        break

    dates = new_dates