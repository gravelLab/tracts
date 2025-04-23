import os
import random


source_dir = "G10_allosomes"

def main():
    # Collect files
    files = [f for f in os.listdir(source_dir)
             if os.path.isfile(os.path.join(source_dir, f)) and '_' in f]
    # Identify individuals
    individuals = set(f.rsplit('_', 1)[0] for f in files)
    # Randomly assign gender
    gender = {ind: random.choice(['male', 'female']) for ind in individuals}
    print("Assigned genders:")
    for ind, g in gender.items():
        print(f"  {ind}: {g}")

    for fname in files:
        ind, hap = fname.rsplit('_', 1)
        g = gender[ind]
        fpath = os.path.join(source_dir, fname)

        # Read all lines
        with open(fpath) as fh:
            lines = [l.rstrip("\n") for l in fh]

        # Extract header + chr1 data
        header = None
        chr1_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            # Header line before chr data
            if line.startswith("Chr") and i+1 < len(lines) and lines[i+1].startswith("1\t"):
                header = line
                j = i+1
                # Collect contiguous chr1 entries
                while j < len(lines) and lines[j].split("\t")[0] == '1':
                    chr1_lines.append(lines[j])
                    j += 1
                break
            i += 1

        if header is None or not chr1_lines:
            print(f"WARNING: Could not find chr1 block in {fname}, skipping.")
            continue

        # Build X block
        x_block = [header]
        for rec in chr1_lines:
            cols = rec.split("\t")
            cols[0] = 'X'
            x_block.append("\t".join(cols))

        # Determine whether to append
        # Male: append only to _A file; Female: append to both
        do_append = (g == 'female') or (g == 'male' and hap.upper() == 'A')
        if do_append:
            with open(fpath, 'a') as fh:
                fh.write("\n" + "\n".join(x_block) + "\n")
            print(f"Appended X block to {fname}")
        else:
            print(f"Skipped {fname} (no X block needed)")


if __name__ == '__main__':
    main()
