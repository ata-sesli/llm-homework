from src.preprocess import extract_model_codes, levenshtein, load_known_words, preprocess


def test_levenshtein():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("", "abc") == 3
    assert levenshtein("oven", "oven") == 0
    assert levenshtein("dishwaser", "dishwasher") == 1


def test_extract_model_codes_normalizes_format():
    assert extract_model_codes("model DW-6534 and wm4410") == ["DW-6534", "WM-4410"]


def test_extract_model_codes_ignores_years_error_codes_and_flights():
    assert extract_model_codes("bought in 2024, error E21, flight DL123 and AB-123") == []


def test_preprocess_removes_codes_and_punctuation():
    result = preprocess("My dishwasher is not draining, model DW-6534!")
    assert result.text == "my dishwasher is not draining model"
    assert result.model_codes == ["DW-6534"]
    assert result.corrections == []


def test_preprocess_folds_turkish_and_maps_product_names():
    result = preprocess("Bulaşık makinem su boşaltmıyor, model DW-6534")
    assert result.text == "dishwasher su bosaltmiyor model"
    assert result.model_codes == ["DW-6534"]


def test_preprocess_collapses_repeated_characters():
    assert preprocess("helloooo!!! pleeeease").text == "helloo pleease"


def test_preprocess_corrects_misspelled_products():
    result = preprocess("my dishwaser and washing mashine are broken")
    assert result.text == "my dishwasher and washing machine are broken"
    assert result.corrections == [("dishwaser", "dishwasher"), ("mashine", "machine")]
    assert preprocess("refridgerator leaking").text == "refrigerator leaking"


def test_preprocess_does_not_correct_known_words():
    assert preprocess("it is over").text == "it is oven"  # no guard -> wrong fix
    assert preprocess("it is over", known_words={"over"}).text == "it is over"


def test_preprocess_skips_short_tokens():
    assert preprocess("the hub is off").text == "the hub is off"


def test_preprocess_requires_same_first_letter():
    # Real false corrections seen on CLINC text before this guard.
    assert preprocess("flashing lights on the ridge").text == "flashing lights on the ridge"


def test_preprocess_keeps_inflected_product_words():
    text = "refrigerated cooktops microwaves washed"
    assert preprocess(text).text == text


def test_load_known_words_uses_frequent_clinc_tokens(tmp_path):
    csv_path = tmp_path / "train.csv"
    csv_path.write_text(
        "text,intent,domain,source\n"
        "is it over,a,x,clinc\n"
        "game over,b,x,clinc\n"
        "over and out,c,x,clinc\n"
        "rare word,d,x,clinc\n"
        "over over over dishwaser,e,appliance_support,appliance\n",
        encoding="utf-8",
    )
    words = load_known_words(csv_path, min_count=3)
    assert "over" in words
    assert "rare" not in words
    assert "dishwaser" not in words
