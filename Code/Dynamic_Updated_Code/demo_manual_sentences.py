from sentence_generator import generate_sentence

pairs = [
    ["Ami", "asa"],
    ["tumi", "jaoa"],
    ["se", "bola"],
    ["Ami", "jaoa"],
    ["tumi", "bola"],
]

for p in pairs:
    print(p, "->", generate_sentence(p))
