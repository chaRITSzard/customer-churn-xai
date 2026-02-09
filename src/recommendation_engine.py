FEATURE_NAME_MAPPING = {
        "cat__InternetService_Fiber optic": "Uses Fiber Optic internet service",
        "cat__MultipleLines_No": "Has only one phone line",
        "cat__StreamingMovies_Yes": "Subscribes to streaming services",
        "cat__Contract_Month-to-month": "On month-to-month contract",
        "cat__TechSupport_No": "Does not have tech support",
        "cat__OnlineSecurity_No": "Does not have online security",
        "num__MonthlyCharges": "High monthly charges",
        "num__tenure": "Short customer tenure"
    }

def translate_features(feature_list):
    readable = []

    for f in feature_list:
        if f in FEATURE_NAME_MAPPING:
            readable.append(FEATURE_NAME_MAPPING[f])
        else:
            readable.append("Customer behavior indicates higher churn risk")
    return readable

def generate_precautions(top_features):

    """
    Mapping risky SHAP Features to business actions
    """
    mapping = {
        "cat__Contract_Month-to-month":
            "Offer discount on annual or 2-year contract",

        "num__MonthlyCharges":
            "Recommend lower-cost plan or provide discount",

        "cat__InternetService_Fiber optic":
            "Investigate service quality or offer alternative plan",

        "cat__MultipleLines_No":
            "Offer bundled multi-line discount",

        "cat__TechSupport_No":
            "Provide free or discounted tech support",

        "cat__OnlineSecurity_No":
            "Bundle online security add-on",

        "cat__PaymentMethod_Electronic check":
            "Encourage enrollment in auto-pay",

        "cat__StreamingMovies_Yes":
            "Offer entertainment bundle discount",

        "num__tenure":
            "Provide loyalty rewards"
    }

    precautions = []

    for feat in top_features:
        if feat in mapping:
            precautions.append(mapping[feat])
        else:
            precautions.append("Offer personalized retention incentive")
    return precautions
