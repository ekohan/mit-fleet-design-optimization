import json
import os

# Parse coverage-summary.json and emit a Markdown table

def main():
    cov_file = os.path.join('coverage', 'coverage-summary.json')
    with open(cov_file) as f:
        data = json.load(f)

    rows = []
    for path, stats in data['files'].items():
        if not path.startswith('src/'):
            continue
        
        # Safely access summary and coverage stats using .get()
        summary_stats = stats.get('summary', {})
        num_statements = summary_stats.get('num_statements', 0)

        if num_statements == 0:
            covered = 100.0
            missing = ""
            gap = 0
        else:
            coverage_stats = stats.get('coverage', {})
            covered = coverage_stats.get('percent', 0.0) # Default to 0 if 'percent' missing
            missing_lines = coverage_stats.get('missing_lines', [])
            missing = ','.join(str(l) for l in missing_lines)
            gap = len(missing_lines)

        rows.append((path, covered, missing, gap))

    print('| Module | Covered % | Missing Lines | Gap Size | New % Goal | Action |')
    print('|--------|----------:|---------------|---------:|-----------:|--------|')
    for path, cov, missing, gap in sorted(rows):
        print(f'| {path} | {cov:6.1f} | {missing} | {gap:8d} |  |  |')

if __name__ == '__main__':
    main() 