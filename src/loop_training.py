import subprocess
from datetime import datetime, timedelta

def str_to_date(s):
    return datetime.strptime(s, "%Y%m%d")

def date_to_str(d):
    return d.strftime("%Y%m%d")

def str_to_date(s):
    return datetime.strptime(s, "%Y%m%d")

def date_to_str(d):
    return d.strftime("%Y%m%d")

# Initial dates and label number
dates = [
    "20240428", "20240622",  # trainFrom, trainTo
    "20240623", "20240629",  # validationFrom, validationTo
    "20240630", "20240706"   # testFrom, testTo
]
label_number = "10"  # Default label number (can be changed)

# End condition
end_date = datetime(2025, 8, 2)

while True:
    # Run the script
    cmd = [
        "python", "stockprice_lstm_tensorflow_regression.py",
        *dates, label_number
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