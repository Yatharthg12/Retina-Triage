CLASS_NAMES = {
    0: "No Apparent Diabetic Retinopathy",
    1: "Mild Non-Proliferative Diabetic Retinopathy",
    2: "Moderate Non-Proliferative Diabetic Retinopathy",
    3: "Severe Non-Proliferative Diabetic Retinopathy",
    4: "Proliferative Diabetic Retinopathy",
}

SHORT_CLASS_NAMES = {
    0: "No apparent DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR",
}

def is_referable(grade: int) -> bool:
    return int(grade) >= 2

def is_high_risk(grade: int) -> bool:
    return int(grade) >= 3

