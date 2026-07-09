# Artificial Intelligence — Assignment Specifications (Session 202605)

## Aims
1. Enable students to analyse and employ appropriate AI techniques to design intelligent systems and solve problems.
2. Enable students to use relevant tools/technology (e.g. Python) to develop intelligent computer programs.

## Learning Outcomes Assessed
- **CLO 2:** Demonstrate AI techniques and strategies to solve a given problem (A3, PLO9).
- **CLO 3:** Produce an AI application using a programming language or other relevant technology (P4, PLO3).

## Overview
- Group assignment: **2–3 members**.
- Two related parts:
  - **Part 1 — Documentation** (40%)
  - **Part 2 — Prototype Development with source code** (60%)
- Team leader compiles and submits both parts before the deadline.
- **Six topic options** — must pick ONE. No two groups may use the same idea/title. Must be original work.
- Achieving basic requirements = Average/Good grade only. **Excellent** grade requires extra effort: new skills, novel ideas, more complex algorithms, big-data handling, and/or excellent reporting + working prototype.

## Our Team's Selected Topic: Option 4 — Natural Language Processing (Sentiment Analysis)

### Requirements for NLP topic
a. **Identify the problem/task** — Sentiment Analysis: is a tweet positive, negative, or neutral (World Cup tweets, multilingual: English/Malay/code-switched).

b. **Background study** covering:
   - The chosen NLP problem/task
   - Significance & real-world applications
   - Common methods/techniques (Bag-of-Words, TF-IDF, word embeddings, transformers)

c. **Data acquisition** — crawl sample data from a forum/social media, OR use a dataset from a reliable source.
   - *(Our case: FIFA World Cup tweets, combined from 2014, 2018, 2022, and 2026.)*
   - 2014, 2018, 2026: scraped (via twscrape)
   - 2022: sourced from a Kaggle dataset
   - Columns across datasets are not identical — plan to drop/align mismatched columns before merging (keep only shared columns).

d. **Preprocessing** to prepare data for analysis:
   - Text cleaning (stop words, punctuation, special characters)
   - Tokenization, stemming, lemmatization
   - Feature extraction (TF-IDF, Word2Vec, BERT embeddings, etc.)
   - Multilingual scope (which languages to include/target — EN/MY/code-switched etc.) — **not yet decided**

e. **Each group member implements a DIFFERENT method.** For sentiment analysis: e.g. Naïve Bayes, SVM, or transformer-based models (BERT/GPT).
   - *(Our team: Ming — BiLSTM, Justin — SVM, Jason — mBERT)*

f. **Compare & evaluate** all models using metrics: **Accuracy, Precision, Recall, F1 Score**.

g. Reference sources (for ideas/datasets):
   - https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html
   - http://adilmoujahid.com/posts/2014/07/twitter-analytics/
   - https://www.yelp.com/dataset
   - https://ai.stanford.edu/~amaas/data/sentiment/

## Submission
- **Deadline:** 28 August 2026 (Week 11, Friday, before 12pm), via Google Classroom.
- Late penalties: 1–3 days late = −10 marks; 4–7 days = −20 marks; >7 days = −100 marks.
- Demo session presenting the prototype: **Week 12–14**. Each member must present their own work and be ready for Q&A.

## Academic Integrity
- Original work only — no copying other groups, no passing off others' ideas as your own.
- Only collaborate within your own team.
- Must complete Appendix A: Plagiarism Statement Form (per student).

## AI Use Disclosure (relevant to using Claude Code!)
- AI tools (ChatGPT, Claude, MidJourney, etc.) are **explicitly allowed** as collaborative learning tools for brainstorming, coding, refining writing.
- **Must disclose all AI use**: include an **AI Disclosure Statement (Appendix B)** with the submission — specify which AI tools were used, what prompts were used, and what steps were taken to verify accuracy/relevance of AI-generated content.
- Students remain **fully responsible** for the accuracy, logic, and integrity of the final work.
- 👉 **Action item:** keep a running log of prompts used with Claude Code for this project — you'll need it for Appendix B.

## Free-Rider Guideline
- Discuss/document individual roles at project start.
- Keep evidence of work (meeting notes, drafts, communication records).
- Unresolved free-riding → Student Free-Rider Report Form (Appendix C) to Course Coordinator/Lecturer/Tutor.

---

# Appendix 1 — Documentation Assessment Rubric (40%)

| CLO | Item | Missing/Unacceptable (0–4) | Poor (5–9) | Accomplished (10–15) | Good (16–20) |
|---|---|---|---|---|---|
| 2 | **Introduction** | Background, problem statement, objectives, significance missing/unclear. No research gap identified. | Present but vague/partial. Limited understanding of research gap. | Present and understandable. Research gap identified and explained. | Clear, comprehensive, well-written. Research gap clearly justified. Objectives/significance fully aligned with study. |
| 2 | **Related Work** | Poorly described or copied without paraphrasing. No evaluation/comparison. | Some description, limited evaluation/comparison. Gaps vaguely mentioned. | Previous studies described, evaluated, compared. Research gaps identified and justified. | Excellent critical analysis and comparison of prior work. Gaps clearly identified with strong justification. |
| 2 | **Methodology** | Missing, irrelevant, or poorly described (system flow, dataset, algorithm, evaluation metrics). | Briefly described but lacks clarity/completeness. Weak algorithm/metrics explanation. | Well-explained with clear system flow, dataset description, algorithm, evaluation metrics. | Comprehensive, logical, clearly presented. Dataset, algorithms, metrics justified and appropriate. |
| 2 | **Results & Discussion** | Missing, unclear, unrelated to objectives. No interpretation/discussion. | Presented but partially unclear/incomplete. Minimal discussion. | Clearly presented, aligned with objectives. Discussion interprets results, addresses implications. | Comprehensive, well-presented, clearly interpreted. Strong discussion of implications and relevance. |
| 2 | **Conclusion & References & Source** | Missing/unclear achievements, limitations, future work. References missing or improperly cited. | Mentioned but limited. References/sources incomplete or improper format. | Achievements, limitations, future work described and explained. References complete, APA cited, datasets/tools acknowledged. | Achievements clearly stated, limitations acknowledged, future improvements thoughtfully proposed. References comprehensive, properly formatted, all datasets/tools fully cited. Strong academic rigor. |

**Final documentation score = (sum of scores / 100) × 40**

---

# Appendix 2 — Prototype Assessment Rubric (60%)

| CLO | Item | Poor (0–4 / 0–8) | Accomplished (5–7 / 9–15) | Good (8–10 / 16–20) |
|---|---|---|---|---|
| 3 | **User Interface / Output (10%)** | Poor/confusing UI, inadequate info/output. Errors present. Layout disorganized. | Adequate info/output generated. Layout organized. | All necessary info/output generated, accurate (minor errors OK). Well-organized layout. |
| 3 | **Programming (20%)** | Many logic errors, no exception handling, over-simplified. Programming skill needs improvement. Minimal/no validations, business rules not validated. | Mostly logical but some steps tedious/complicated. Acceptable algorithm complexity. Minimal validations provided. | Correct, logical flow; exceptions handled well. High-level complex algorithms and programming skill. Thorough validations. All important business rules validated. |
| 3 | **Degree of Completion (10%)** | Basic requirements not fulfilled, much left to do. | All required features present within scope, but some simplified, or 1–2 features missing. | All required features present within or beyond required scope. |
| 3 | **System Implementation (10%)** | Produces enormous errors/faults/incorrect results. Different system design unrelated to proposal. | System runs with minor errors. Conforms to most of proposed design, some parts different. | No bugs during demo. Fully conforms to proposed system design. |
| 3 | **Presentation & On-the-Spot Coding (10%)** | Unclear about own work, doesn't know where to find source code. | Knows code whereabouts but may be unclear why work was done a certain way. | Clear about every piece of work done. |

**Final prototype score = sum of scores (already out of 60)**

---

# Key Deliverables Checklist for Our NLP (Sentiment Analysis) Project

- [ ] Background study: sentiment analysis problem, significance, methods (BoW, TF-IDF, embeddings, transformers)
- [ ] Related work: comparison of prior sentiment analysis studies (esp. multilingual/code-switched, mBERT vs SVM)
- [ ] Dataset: FIFA World Cup tweets (2014, 2018, 2022, 2026 — merged; 2014/2018/2026 scraped, 2022 from Kaggle)
- [ ] Align columns across all four datasets before merging (drop non-shared columns)
- [ ] Decide multilingual scope (which languages to include/target) — **pending decision**
- [ ] Preprocessing pipeline: cleaning, tokenization, stemming/lemmatization, feature extraction
- [ ] Three distinct models (one per member): BiLSTM (Ming), SVM (Justin), mBERT (Jason)
- [ ] Evaluation & comparison: Accuracy, Precision, Recall, F1 across all three models
- [ ] Working prototype with UI/output display
- [ ] Documentation: Introduction, Related Work, Methodology, Results & Discussion, Conclusion, References (APA)
- [ ] Appendix A: Plagiarism Statement Form (per member)
- [ ] Appendix B: AI Disclosure Statement (log tools/prompts used + verification steps)
- [ ] Appendix C: Free-Rider Report Form (only if needed)
- [ ] Each member ready to individually present + do on-the-spot code walkthrough (Week 12–14 demo)

**Deadline: 28 August 2026, before 12pm, via Google Classroom.**