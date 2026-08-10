import warnings
from rag.loader import load_documents


def main():
    # ------------------------------------------------------------------
    # AI-060: Document Loading (fully offline — no API calls)
    # ------------------------------------------------------------------
    print("=" * 55)
    print("AI-060: Document Loading Tests (Offline)")
    print("=" * 55)

    # Test 1: Happy path — load from the docs/ directory
    print("\n--- Test 1: Load from docs/ directory ---")
    docs = load_documents("docs")
    if docs:
        for doc in docs:
            print(f"  Source   : {doc.metadata['source']}")
            print(f"  Chars    : {len(doc.content)}")
            print(f"  Preview  : {doc.content[:80].strip()!r}...")
            print()
        print(f"  Total documents loaded: {len(docs)}")
    else:
        print("  [WARN] No documents loaded.")

    # Test 2: Failure path — nonexistent directory
    print("\n--- Test 2: Nonexistent directory ---")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = load_documents("nonexistent_dir")
    if caught:
        print(f"  Warning caught: {caught[0].message}")
    print(f"  Returned: {result}")
    assert result == [], "Expected empty list for missing directory"
    print("  PASS — returned empty list as expected.")

    print("\n" + "=" * 55)
    print("AI-060 tests complete. No API calls were made.")
    print("=" * 55)


if __name__ == "__main__":
    main()
