# default_data.py - provides the default presentation structure

def get_default_presentation():
    """Return a dictionary representing a 12‑slide pitch deck.
    Each slide is a dict with keys:
        - "title": slide title
        - "content": list of paragraph strings or bullet points
    The structure is deliberately simple – the generators only need
    a title and a list of strings to render.
    """
    return {
        "metadata": {
            "project_name": "Botnet Detection Using Machine Learning",
            "subtitle": "An Autoencoder–XGBoost Hybrid Approach",
            "team": ["Prajwal NP", "Kiran Kumar PN", "Puneeth Raj R"],
            "platform": "IBM LinuxONE Community Cloud",
            "domain": "Artificial Intelligence • Machine Learning • Cybersecurity",
        },
        "slides": [
            {
                "title": "Title",
                "content": [
                    "Botnet Detection Using Machine Learning",
                    "An Autoencoder–XGBoost Hybrid Approach",
                    "Team: Prajwal NP, Kiran Kumar PN, Puneeth Raj R",
                    "Platform: IBM LinuxONE Community Cloud",
                    "Domain: AI • ML • Cybersecurity",
                ],
            },
            {"title": "Problem Statement", "content": ["Current botnet detection methods suffer from high false‑positive rates and limited scalability."]},
            {"title": "Proposed Solution", "content": ["Hybrid approach: Autoencoder for feature extraction + XGBoost classifier."]},
            {"title": "Data Sources", "content": ["CTIJI ormaI_Traffic.csv – normal traffic", "CTU13_Attack_Traffic.csv – attack traffic"]},
            {"title": "Methodology", "content": ["1. Data preprocessing", "2. Autoencoder training", "3. Feature extraction", "4. XGBoost classification", "5. Evaluation"]},
            {"title": "Results", "content": ["Achieved > 95% detection accuracy on test set."]},
            {"title": "Architecture Diagram", "content": ["(Insert diagram placeholder)\n"]},
            {"title": "Comparison with Baselines", "content": ["Baseline models: RandomForest, IsolationForest – lower performance."]},
            {"title": "Deployment", "content": ["Containerized service ready for IBM LinuxONE."]},
            {"title": "Future Work", "content": ["Live streaming detection, adversarial robustness."]},
            {"title": "Conclusion", "content": ["Hybrid model provides robust, scalable botnet detection."]},
            {"title": "Thank You", "content": ["Contact: your.email@example.com"]},
        ],
    }
