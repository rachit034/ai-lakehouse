from core.nl_cache_key import literal_signature as sig


def test_different_years_differ():
    assert sig("What was revenue in 2024?") != sig("What was revenue in 2023?")


def test_direction_differs():
    assert sig("Show the top 5 customers by spend") != sig("Show the bottom 5 customers by spend")


def test_proper_nouns_differ():
    assert sig("How many orders came from Germany?") != sig("How many orders came from France?")


def test_quoted_values_differ():
    assert sig("orders with status 'shipped'") != sig("orders with status 'cancelled'")


def test_negation_differs():
    assert sig("customers who ordered electronics") != sig("customers who never ordered electronics")


def test_paraphrases_match():
    assert sig("find me the customer with maximum orders") == sig("which customer has the most orders")
    assert sig("Which customer has the highest lifetime value?") == sig("customer with max lifetime value")
