# Example terminal commands used for testing:
#   py classifier.py authorlist.txt
#   py classifier.py authorlist.txt --order 1
#   py classifier.py authorlist.txt --order 2
#   py classifier.py authorlist.txt --order 3
#   py classifier.py authorlist.txt -test austen_test_sents.txt
#   py classifier.py authorlist.txt -test dickens_test_sents.txt
#   py classifier.py authorlist.txt -test tolstoy_test_sents.txt
#   py classifier.py authorlist.txt -test wilde_test_sents.txt
#   py classifier.py authorlist.txt -test testfile.txt
import argparse, math, os, random
from collections import Counter
import tiktoken
class Tokenizer:
    def __init__(self, name="o200k_base"):
        self.tokenizer = tiktoken.get_encoding(name)
    def encode(self, text): #change to number list
        return self.tokenizer.encode(text)
#helper function
def read_lines(path):
    f = open(path, "r", encoding="utf-8", errors="ignore")
    lines = []
    for line in f:
        lines.append(line.rstrip("\n"))
    f.close()
    return lines
def read_text(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()
def derive_name(filename):
    base = os.path.basename(filename)
    return base.split("_train")[0]
def sentences_from_text(text):
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            lines.append(line)
    return lines
def split_train_dev(sentences, dev_ratio=0.1, seed=42):
    rand = random.Random(seed)
    num_sentences = len(sentences)
    indices = list(range(num_sentences))
    rand.shuffle(indices)
    devn = int(num_sentences * dev_ratio)
    if devn < 1:
        devn = 1
    dev_indices = set(indices[:devn])
    train = []
    dev = []
    for i, sentence in enumerate(sentences):
        if i in dev_indices:
            dev.append(sentence)
        else:
            train.append(sentence)
    return train, dev
# AI assistance (ChatGPT) was used for conceptual clarification and minor code review.
class InterpNGramLM:
    def __init__(self, order=3, add_k=0.1, lambdas=(0.2, 0.3, 0.5)):
        self.order = order
        self.add_k = add_k
        # interpolation weights
        self.l1, self.l2, self.l3 = self._make_lambdas(order, lambdas)
        # count tables
        self.uni = Counter()
        self.bi = Counter()
        self.tri = Counter()
        self.bi_ctx = Counter()     # count(w_prev)
        self.tri_ctx = Counter()    # count(w2, w1)
        # vocab info
        self.vocab = set()
        self.V = 0
        # sentence boundary tokens
        self.bos_id = -1
        self.eos_id = -2

    def _make_lambdas(self, order, lambdas):
        l1, l2, l3 = lambdas
        if order == 1:
            return 1.0, 0.0, 0.0
        if order == 2:
            s = l1 + l2
            if s == 0:
                return 1.0, 0.0, 0.0
            return l1 / s, l2 / s, 0.0
        s = l1 + l2 + l3
        if s == 0:
            return 1.0, 0.0, 0.0
        return l1 / s, l2 / s, l3 / s

    def fit(self, sentences_token_ids):
        for sent in sentences_token_ids:
            seq = [self.bos_id, self.bos_id] + sent + [self.eos_id]

            for tok in sent:
                self.vocab.add(tok)
            self.vocab.add(self.eos_id)

            for t in range(2, len(seq)):
                w2, w1, w = seq[t - 2], seq[t - 1], seq[t]
                self.uni[w] += 1
                self.bi[(w1, w)] += 1
                self.bi_ctx[w1] += 1
                self.tri[(w2, w1, w)] += 1
                self.tri_ctx[(w2, w1)] += 1
        self.V = max(1, len(self.vocab))
    def p_uni(self, w):
        total = sum(self.uni.values())
        return (self.uni[w] + self.add_k) / (total + self.add_k * self.V)
    def p_bi(self, w_prev, w):
        denom = self.bi_ctx[w_prev]
        return (self.bi[(w_prev, w)] + self.add_k) / (denom + self.add_k * self.V)
    def p_tri(self, w2, w1, w):
        denom = self.tri_ctx[(w2, w1)]
        return (self.tri[(w2, w1, w)] + self.add_k) / (denom + self.add_k * self.V)

    def log_prob(self, sent):
        seq = [self.bos_id, self.bos_id] + sent + [self.eos_id]
        lp = 0.0

        for t in range(2, len(seq)):
            w2, w1, w = seq[t - 2], seq[t - 1], seq[t]
            p = self.l1 * self.p_uni(w)
            if self.order >= 2 and self.l2 > 0.0:
                p += self.l2 * self.p_bi(w1, w)
            if self.order >= 3 and self.l3 > 0.0:
                p += self.l3 * self.p_tri(w2, w1, w)
            if p <= 0.0:
                p = 1e-12
            lp += math.log(p)
        return lp

    def perplexity(self, sent):
        N = max(1, len(sent) + 1)
        return math.exp(-self.log_prob(sent) / N)
def tokenize_sentences(tok, sentences):
    return [tok.encode(s) for s in sentences]
def train_author_models(author_files, tok, use_full_data, order, add_k, lambdas):
    models = {}
    dev_data = {}

    for path in author_files:
        author = derive_name(path)
        sentences = sentences_from_text(read_text(path))

        if use_full_data:
            train_sents = sentences
            dev_sents = []
        else:
            train_sents, dev_sents = split_train_dev(sentences, 0.1)

        train_ids = tokenize_sentences(tok, train_sents)
        dev_ids = tokenize_sentences(tok, dev_sents)

        lm = InterpNGramLM(order=order, add_k=add_k, lambdas=lambdas)
        lm.fit(train_ids)

        models[author] = lm
        dev_data[author] = dev_ids
    return models, dev_data
def predict(models, sent_ids):
    best_author, best_pp = None, float("inf")
    for author, lm in models.items():
        pp = lm.perplexity(sent_ids)
        if pp < best_pp:
            best_author, best_pp = author, pp
    assert best_author is not None
    return best_author
def eval_dev(models, dev_data):
    print("Results on dev set:")
    for author in sorted(dev_data.keys()):
        sents = dev_data[author]
        if not sents:
            print(f"{author}\t(no dev data)")
            continue
        correct = sum(1 for s in sents if predict(models, s) == author)
        acc = 100.0 * correct / len(sents)
        print(f"{author}\t{acc:.1f}% correct")
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("authorlist", help="File containing training filenames (one per line).")
    p.add_argument("-test", dest="testfile", default=None, help="Test file (each line is a sentence).")
    p.add_argument("--encoding", default="o200k_base")
    p.add_argument("--order", type=int, default=3, choices=[1, 2, 3])
    p.add_argument("--add_k", type=float, default=0.1)
    p.add_argument("--l1", type=float, default=0.2)
    p.add_argument("--l2", type=float, default=0.3)
    p.add_argument("--l3", type=float, default=0.5)
    return p.parse_args()

def main():
    args = parse_args()
    author_files = [ln.strip() for ln in read_lines(args.authorlist) if ln.strip()]
    tok = Tokenizer(args.encoding)
    lambdas = (args.l1, args.l2, args.l3)
    print("training... (this may take a while)")
    use_full = args.testfile is not None
    models, dev_data = train_author_models(
        author_files=author_files,
        tok=tok,
        use_full_data=use_full,
        order=args.order,
        add_k=args.add_k,
        lambdas=lambdas,
    )
    if args.testfile is None:
        eval_dev(models, dev_data)
    else:
        for line in read_lines(args.testfile):
            sent = line.strip()
            if not sent:
                continue
            sent_ids = tok.encode(sent)
            print(predict(models, sent_ids))

if __name__ == "__main__":
    main()