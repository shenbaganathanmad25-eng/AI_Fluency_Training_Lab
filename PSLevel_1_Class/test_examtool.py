from task_agent import check_exam_eligibility

for attendance in [82, 70, 50, 120]:
    result = check_exam_eligibility.invoke(
        {"attendance_percent": attendance}
    )
    print(attendance, "->", result)