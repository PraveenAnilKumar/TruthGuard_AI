with open("d:/TruthGuard_AI/app.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = lines[:1491] + lines[1506:]

with open("d:/TruthGuard_AI/app.py", "w", encoding="utf-8", newline='') as f:
    f.writelines(new_lines)

print("Done")
