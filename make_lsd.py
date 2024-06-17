# Bachelorarbeit
# Sara Derakhshani
# Matrikelnummer: 792483
# Abgabe: 01.07.2022 

from csv import writer
from re import sub
import configparser


def extract_polarity(polarity: str):
    """Extract the polarity label. Returns the polarity without unnecessary characters."""
    polarity = polarity[1:].strip()
    if "#" in polarity:
        polarity = polarity.split("#")[0]
    return polarity


def create_lex_entry(pattern: str):
    """Create an entry with meta info for the new LSD.
    Returns a list containing the lexicon entry, the number of tokens, 0 or 1 (prefix or no prefix)."""
    pattern = pattern.strip()
    tokens = pattern.split()
    is_prefix = False
    if pattern.endswith("*"):
        is_prefix = True
        pattern = sub('\*', '', pattern)
    return [pattern, len(tokens), int(is_prefix)]


def main(unprocessed_lsd, unprocessed_lsd_neg, lsd):
    polarity = None
    # Count entries for progress bar
    n_lsd_entries = sum([1 for n in open(unprocessed_lsd, "r")])
    n_lsd_neg_entries = sum([1 for m in open(unprocessed_lsd_neg, "r")])
    total_n_entries = n_lsd_entries + n_lsd_neg_entries
    # Iterate LSD and negated LSD and add entry to new lexicon file
    with open(lsd, "w", encoding="utf-8") as out_f:
        out_writer = writer(out_f, delimiter='\t')
        out_writer.writerow(["lexEntry", "nrOfTokens", "isPrefix", "polarity"])
        with open(unprocessed_lsd, "r", encoding="utf-8") as lex_f:
            for line in lex_f:
                # Get polarity
                if line.startswith("+"):
                    polarity = extract_polarity(line)
                    continue
                # Skip this entry
                elif "unite" in line:
                    continue
                new_entry = create_lex_entry(line)
                new_entry.append(polarity)
                out_writer.writerow(new_entry)
        lex_f.close()
        with open(unprocessed_lsd_neg, "r", encoding="utf-8") as lex_neg_f:
            for line in lex_neg_f:
                # Get polarity
                if line.startswith("+"):
                    polarity = extract_polarity(line)
                    continue
                # Skip this entry
                elif "unite" in line:
                    continue
                new_entry = create_lex_entry(line)
                new_entry.append(polarity)
                out_writer.writerow(new_entry)
        lex_neg_f.close()
    out_f.close()
    print(f"\n Lexicon successfully created and saved in {lsd} \n")


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config.ini")
    unprocessed_lsd = config["LEXICODER"]["UNPROCESSED_LSD"]
    unprocessed_lsd_neg = config["LEXICODER"]["UNPROCESSED_LSD_NEG"]
    lsd = config["LEXICODER"]["LSD"]

    main(unprocessed_lsd, unprocessed_lsd_neg, lsd)