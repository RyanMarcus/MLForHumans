def read_csv(path):
    with open(path) as f:
        first, *last = f
        headers = first.replace('\n', "").strip().split(",")
        last = list(map(lambda x: x.replace("\n", "").replace("?", "0.0").strip().split(","), last))
        return headers, list(last)
            
