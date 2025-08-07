import subprocess
from datetime import datetime, timedelta

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
label_number = "5"  # Default label number for averaging (must be odd: 5, 7, 9, 11, 13, etc.)

# End condition
end_date = datetime(2025, 8, 2)

while True:
    # Run the averaging script
    cmd = [
        "python", "stockprice_lstm_tensorflow_regression_avg.py",
        *dates, label_number
    ]
    print("Running:", " ".join(cmd))
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully completed training for label {label_number} with dates {dates}")
    except subprocess.CalledProcessError as e:
        print(f"Error running training script: {e}")
        print("Continuing to next iteration...")

    # Shift all dates by 7 days
    new_dates = [date_to_str(str_to_date(d) + timedelta(days=7)) for d in dates]

    # Check if testTo (last date) exceeds end_date
    if str_to_date(new_dates[-1]) > end_date:
        print("Reached end date. Stopping.")
        break

    dates = new_dates

print("Loop training completed!")
