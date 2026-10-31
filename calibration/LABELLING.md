# Labelling the 100 answers

Open `calibration/labels.csv` in a spreadsheet. Each row is one question from the SQuAD 2.0 development set,
the gold answer(s), and the answer a small local model (`qwen3:8b`) gave after reading the paragraph.

Fill in the last column, `label`, with exactly one of:

| label | meaning |
|---|---|
| `correct` | The answer says what the gold answer says. Extra words are fine if nothing in them is wrong. |
| `partly` | Right idea but incomplete, too vague, or with a wrong extra detail. |
| `wrong` | A different answer, or no real answer. |

Unanswerable questions show `(unanswerable)` as the gold answer. The model is told to reply `unanswerable`
when the paragraph does not contain the answer:

- it says `unanswerable` (or clearly says the paragraph does not say): `correct`
- it gives an answer anyway: `wrong`

Judge only against the gold answer, not against what you know to be true. Do not look at any judge's score;
the labels must not be influenced by it. Do not change any other column.

Save the file as CSV (UTF-8 if the spreadsheet asks), in the same place, and tell Claude. It will check the
labels (`calibration/prepare.py import-labels`), and only then run the hosted judges.

`labels.csv` holds SQuAD text (CC BY-SA 4.0), so it is git-ignored. What gets committed is
`calibration/labels_public.csv`: just `id,label`.
