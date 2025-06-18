# 📘 Project Title


## 🧑‍💻 Team Members

- Fabian Weterbeek – fabian.westerbeek@student.uva.nl
- Bhavesh Sood – bhavesh.sood@student.uva.nl 
- Kshitiz Sharma – kshitiz.sharma2@student.uva.nl
- Maxim Voronin - maxim.voronin@student.uva.nl

## 👥 Supervising TAs
- Yubao Tang (Main Supervisor)
- Owen de Jong (Co-supervisor)


---

## 🧾 Project Abstract
_Provide a concise summary of your project, including the type of recommender system you're building, the key techniques used, and a brief two sentence summary of results._

---

## 📊 Summary of Results


### Reproducability 

_Summarize your key reproducability findings in bullet points._

### Extensions

_Summarize your key findings about the extensions you implemented in bullet points._

---

## 🛠️ Task Definition
_Define the recommendation task you are solving (e.g., sequential, generative, content-based, collaborative, ranking, etc.). Clearly describe inputs and outputs._

---

## 📂 Datasets

_Provide the following for all datasets, including the attributes you are considering to measure things like item fairness (for example)_:

- [ ] [Dataset Name](Link-to-dataset-DOI-or-URL)
  - [ ] Pre-processing: e.g., Removed items with fewer than 5 interactions, and users with fewer than 5 interactions
  - [ ] Subsets considered: e.g., Cold Start (5-10 items)
  - [ ] Dataset size: # users, # items, sparsity:
  - [ ] Attributes for user fairness (only include if used):
  - [ ] Attributes for item fairness (only include if used):
  - [ ] Attributes for group fairness (only include if used):
  - [ ] Other attributes (only include if used):

---

## 📏 Metrics

_Explain why these metrics are appropriate for your recommendation task and what they are measuring briefly._

- [ ] Metric #1
  - [ ] Description:
- [ ] Intra-List Diversity (ILD)
  - [ ] Description: Measures the diversity of recommendations by calculating the average pairwise cosine distance between content embeddings of recommended items. Higher values indicate more diverse recommendations.



---

## 🔬 Baselines & Methods

_Describe each baseline, primary methods, and how they are implemented. Mention tools/frameworks used (e.g., Surprise, LightFM, RecBole, PyTorch)._
Describe each baseline
- [ ] [Baseline 1](Link-to-reference)
- [ ] [Baseline 2](Link-to-reference)


### 🧠 High-Level Description of Method

_Explain your approach in simple terms. Describe your model pipeline: data input → embedding/representation → prediction → ranking. Discuss design choices, such as use of embeddings, neural networks, or attention mechanisms._

---

## 🌱 Proposed Extensions

_List & briefly describe the extensions that you made to the original method, including extending evaluation e.g., other metrics or new datasets considered._

- Semantic ID to Content Embedding Lookup Table: Implemented a lookup table that maps semantic IDs to their corresponding content embeddings, enabling efficient retrieval and additional metrics.
- Intra-List Diversity (ILD) Metric: Added a new evaluation metric that measures the diversity of recommendation lists based on content embeddings, complementing existing Gini coefficient diversity measurements.




