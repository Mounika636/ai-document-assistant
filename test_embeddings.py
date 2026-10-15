
from embeddings import generate_embeddings


def test_embedding_dimensions():
    vectors = generate_embeddings([
        "Artificial intelligence is transforming software.",
        "Machine learning helps computers recognize patterns.",
    ])

    assert len(vectors) == 2
    assert len(vectors[0]) == 384
    assert len(vectors[1]) == 384


def test_empty_input():
    assert generate_embeddings([]) == []


def test_semantic_similarity():
    vectors = generate_embeddings([
        "I enjoy learning about artificial intelligence.",
        "I like studying AI and machine learning.",
        "The refrigerator keeps food cold.",
    ])

    def cosine_similarity(a, b):
        return sum(x * y for x, y in zip(a, b))

    related_similarity = cosine_similarity(vectors[0], vectors[1])
    unrelated_similarity = cosine_similarity(vectors[0], vectors[2])

    assert related_similarity > unrelated_similarity
