# Labeling Guide (v1)

## English

## Objective
We label the **search relevance** of retrieved fragments with respect to the query.  
We do not evaluate the quality of a full or long answer, only whether the fragment is **useful** in answering the query.

For each query, the candidate set is obtained from BM25:  
- we take the **top 15 fragments** (highest BM25 scores),  
- plus **5 random fragments** from ranks 16–100.  

Thus, each query has 20 candidates to label, covering both the top of the ranking and more varied control fragments.

---

## Relevance labels
- **2 (Fully relevant)**  
  The fragment defines or directly explains the concept/metaphor in the query.  
  → clear, direct answer, may include figurative/literal explanation.

- **1 (Partially relevant / supporting)**  
  The fragment doesn’t provide a clear definition but helps:  
  – offers examples, contexts, applications.  
  – complements understanding but isn’t sufficient alone.

- **0 (Not relevant)**  
  The fragment does not answer the query.  
  – different topic, literal confusion, or only tangential.

---

## Optional tags (for fine analysis)
- `metaphor-definition` – the fragment gives a definition of the metaphor.  
- `metaphor-usage` – the fragment shows the metaphor in use (e.g., in a financial article).  
- `literal-confounder` – the term appears in a literal, not figurative sense.  
- `domain-shift` – relevant but from another domain (e.g., medical instead of economic).

---

## Fine rules
1. If a fragment mixes figurative and literal sense → decide the label by **usefulness for the query**.  
2. For cross-lingual, **do not penalize the language**: if the content answers, it’s relevant even in another language.  
3. If both definition and examples exist → mark as **2 (definition has priority)**.  
4. If unclear → prefer **1** over 0, to preserve contextual information.  
5. Note tags only if obvious; don’t invent.

---

## Calibrated examples

### Romanian
1. **Query:** “What does «vânturi potrivnice» mean in economics?”  
   **Fragment:** “The term ‘headwinds’ describes factors that slow economic growth.”  
   **Label:** 2, Tags: [metaphor-definition]

2. **Query:** “Explain the expression «piața a digerat știrea».”  
   **Fragment:** “Investors reacted quickly to the publication of the financial report.”  
   **Label:** 1, Tags: [metaphor-usage]

3. **Query:** “What is ‘liquidity’ in capital markets?”  
   **Fragment:** “Liquidity is the amount of water in the soil...”  
   **Label:** 0, Tags: [literal-confounder]

4. **Query:** “What does ‘piață fierbinte’ mean?”  
   **Fragment:** “A hot market in real estate refers to high demand and rising prices.”  
   **Label:** 2, Tags: [metaphor-definition]

5. **Query:** “Define ‘runway’ in business.”  
   **Fragment:** “The company has enough capital for 12 months of operation — this is its runway.”  
   **Label:** 2, Tags: [metaphor-definition]

### English
6. **Query:** “Explain ‘short squeeze’ in simple terms.”  
   **Fragment:** “A short squeeze happens when short sellers are forced to buy back shares as prices rise.”  
   **Label:** 2, Tags: [metaphor-definition]

7. **Query:** “What does ‘digest the news’ mean?”  
   **Fragment:** “The market quickly absorbed the announcement.”  
   **Label:** 2, Tags: [metaphor-usage]

8. **Query:** “Define liquidity trap.”  
   **Fragment:** “A liquidity trap occurs when interest rates are low and savings rates are high, rendering monetary policy ineffective.”  
   **Label:** 2

9. **Query:** “What is a rally in markets?”  
   **Fragment:** “After the crisis, the stock market rally lifted major indexes by 10%.”  
   **Label:** 1, Tags: [metaphor-usage]

10. **Query:** “What are headwinds for the economy?”  
    **Fragment:** “Strong consumer demand supported growth, but rising energy prices created headwinds.”  
    **Label:** 2, Tags: [metaphor-definition, metaphor-usage]

---

## Phase 1 — Initial calibration (overlap zones)

**Goal:** align interpretations before starting full labeling.

- Each annotator labels **40 queries**: **10** per category (en_en, en_ro, ro_en, ro_ro).  
- All use the **same 40 QID**, to allow direct comparison.

**Evaluation:**  
- Compute **Cohen’s κ (simple, nominal)** between each pair (Liana–Andrei, Andrei–Gaby, Liana–Gaby).  
- Report both overall κ and κ on the cross-lingual subset (en↔ro).  
- Targets:  
  - κ ≥ 0.60 overall  
  - κ ≥ 0.55 on the cross-lingual subset  

**If thresholds are not reached:**  
- discuss disagreement cases (e.g., 2 vs 1, 2 vs 0),  
- adjust the rules in the guide,  

---

## Calibration, allocation, and quality control

### Slices (100 each)
- **en_en_###** — EN→EN  
- **en_ro_###** — EN→RO (cross-lingual)  
- **ro_en_###** — RO→EN (cross-lingual)  
- **ro_ro_###** — RO→RO

### Allocation per slice (100 items)

Equal distribution: **40–40–40**.  
The split scheme is **circular, with overlap on all pairs, but no triple-overlap**:

- **Gaby [A] (40 QID):**  
  - A∩B (7): `001–007`  
  - A-only (27): `008–034`  
  - A∩C (6): `035–040`

- **Liana [B] (40 QID):**  
  - A∩B (7): `001–007`  
  - B-only (26): `041–066`  
  - B∩C (7): `067–073`

- **Andrei [C] (40 QID):**  
  - A∩C (6): `035–040`  
  - B∩C (7): `067–073`  
  - C-only (27): `074–100`

**Properties:**  
- |A|=|B|=|C|=40; |A ∪ B ∪ C|=100.  
- Pairwise overlaps: A∩B=7, B∩C=7, A∩C=6.  
- No triple-overlap.  
- Each QID is labeled by 1 or 2 people.

### Output convention (JSONL format)
```json
{"qid":"ro_0001","chunk_id":"fjo3HaoKLs40I","rank":1,"score":13.2,"from":"bm25_en_mt","tags":["metaphor-definition"]}
```

## Romanian

## Obiectiv
Etichetăm **relevanța pentru căutare** a fragmentelor recuperate de sistem, în raport cu întrebarea (query).
Nu evaluăm calitatea unui răspuns complet sau lung, ci dacă fragmentul este **util** pentru a răspunde la întrebare.

Pentru fiecare query, setul de candidați este obținut din BM25:  
- luăm **primele 15 fragmente** (cele mai relevante după scorul BM25),  
- și **5 fragmente aleatoare** din intervalul rank 16–100.  

Astfel, fiecare query are 20 candidați de etichetat, acoperind atât vârful clasamentului, cât și fragmente mai variate pentru control.

---

## Etichete de relevanță
- **2 (Relevant complet)**  
  Fragmentul definește sau explică direct conceptul/metafora din întrebare.  
  → răspuns direct, clar, poate include și explicații figurative/literale.

- **1 (Parțial relevant / de sprijin)**  
  Fragmentul nu oferă definiția clară, dar ajută:  
  – oferă exemple de utilizare, contexte, aplicații.  
  – completează înțelegerea dar nu e suficient singur.

- **0 (Nerelevant)**  
  Fragmentul nu răspunde la întrebare.  
  – alt subiect, confuzie literală, sau doar tangential.

---

## Tag-uri opționale (pentru analiză fină)
- `metaphor-definition` – fragmentul dă definiția metaforei.
- `metaphor-usage` – fragmentul arată metafora în uz (ex.: într-un articol financiar).
- `literal-confounder` – termenul apare în sens literal, nu figurativ.
- `domain-shift` – relevant, dar din alt domeniu (ex. medical în loc de economic).

---

## Reguli fine
1. Dacă un fragment amestecă sens figurat și literal → eticheta se decide după **utilitatea pentru query**.
2. La cross-lingual, **nu penaliza limba**: dacă conținutul răspunde, e relevant chiar dacă e în altă limbă.
3. Dacă există atât definiție cât și exemple → marchează ca **2 (definiția are prioritate)**.
4. Dacă nu e clar → alege mai degrabă **1** decât 0, pentru a păstra informația de context.
5. Notează tag-urile doar dacă sunt evidente; nu inventa.

---

## Exemple calibrate

### Română
1. **Query:** „Ce înseamnă «vânturi potrivnice» în economie?”  
   **Fragment:** „Termenul ‘headwinds’ descrie factori care încetinesc creșterea economică.”  
   **Label:** 2, Tags: [metaphor-definition]

2. **Query:** „Explică expresia «piața a digerat știrea».”  
   **Fragment:** „Investitorii au reacționat rapid la publicarea raportului financiar.”  
   **Label:** 1, Tags: [metaphor-usage]

3. **Query:** „Ce este ‘lichiditatea’ pe piața de capital?”  
   **Fragment:** „Lichiditatea este cantitatea de apă din sol...”  
   **Label:** 0, Tags: [literal-confounder]

4. **Query:** „Ce înseamnă ‘piață fierbinte’?”  
   **Fragment:** „Hot market în imobiliare se referă la o cerere mare și prețuri crescute.”  
   **Label:** 2, Tags: [metaphor-definition]

5. **Query:** „Definește ‘runway’ în business.”  
   **Fragment:** „Compania are capital suficient pentru 12 luni de operare — acesta este runway-ul.”  
   **Label:** 2, Tags: [metaphor-definition]

### Engleză
6. **Query:** “Explain ‘short squeeze’ in simple terms.”  
   **Fragment:** “A short squeeze happens when short sellers are forced to buy back shares as prices rise.”  
   **Label:** 2, Tags: [metaphor-definition]

7. **Query:** “What does ‘digest the news’ mean?”  
   **Fragment:** “The market quickly absorbed the announcement.”  
   **Label:** 2, Tags: [metaphor-usage]

8. **Query:** “Define liquidity trap.”  
   **Fragment:** “A liquidity trap occurs when interest rates are low and savings rates are high, rendering monetary policy ineffective.”  
   **Label:** 2

9. **Query:** “What is a rally in markets?”  
   **Fragment:** “After the crisis, the stock market rally lifted major indexes by 10%.”  
   **Label:** 1, Tags: [metaphor-usage]

10. **Query:** “What are headwinds for the economy?”  
    **Fragment:** “Strong consumer demand supported growth, but rising energy prices created headwinds.”  
    **Label:** 2, Tags: [metaphor-definition, metaphor-usage]

---

## Faza 1 — Calibrare inițială (zonele de suprapunere)

**Scop:** aliniem interpretările înainte să pornim etichetarea completă.

- Fiecare anotorator etichetează **40 întrebări**: câte **10** per categorie (en_en, en_ro, ro_en, ro_ro).  
- Toți folosesc **aceleași 40 QID**, pentru a putea compara direct.

**Evaluare:**  
- Se calculează **Cohen’s κ simplu (nominal)** între fiecare pereche (Liana–Andrei, Andrei–Gaby, Liana–Gaby).  
- Se raportează atât κ global, cât și κ pe subsetul cross-lingual (en↔ro).  
- Ținte:  
  - κ ≥ 0.60 global  
  - κ ≥ 0.55 pe subsetul cross-lingual  

**Dacă pragurile nu sunt atinse:**  
- discutăm cazurile de dezacord (ex. 2 vs 1, 2 vs 0),  
- ajustăm regulile din ghid,  

---

## Calibrare, alocare și control calitate

### Slices (câte 100 fiecare)
- **en_en_###** — EN→EN  
- **en_ro_###** — EN→RO (cross-lingual)  
- **ro_en_###** — RO→EN (cross-lingual)  
- **ro_ro_###** — RO→RO

### Alocare per slice (100 itemi)

Distribuție egală: **40–40–40**.  
Schema de împărțire este **circulară, cu overlap pe toate perechile, dar fără triple-overlap**:

- **Gaby [A] (40 QID):**  
  - A∩B (7): `001–007`  
  - A-only (27): `008–034`  
  - A∩C (6): `035–040`

- **Liana [B] (40 QID):**  
  - A∩B (7): `001–007`  
  - B-only (26): `041–066`  
  - B∩C (7): `067–073`

- **Andrei [C] (40 QID):**  
  - A∩C (6): `035–040`  
  - B∩C (7): `067–073`  
  - C-only (27): `074–100`

**Proprietăți:**  
- |A|=|B|=|C|=40; |A ∪ B ∪ C|=100.  
- Overlap perechi: A∩B=7, B∩C=7, A∩C=6.  
- Nu există triple-overlap.  
- Fiecare QID este etichetat de 1 sau 2 persoane.  

### Convenție output (format JSONL)
```json
{"qid":"ro_0001","chunk_id":"fjo3HaoKLs40I","rank":1,"score":13.2,"from":"bm25_en_mt","tags":["metaphor-definition"]}
```
