import json

prof_file = './prof.json'
with open(prof_file, 'rb') as f:
    prof = json.load(f)

sorted_prof = sorted(prof, key=lambda d: d.get('percentage', 0), reverse=True)
for i, item in enumerate(sorted_prof):
    if i < 200:
        print(item)
