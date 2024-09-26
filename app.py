import gradio as gr
import torch
import transformers

model: transformers.AlbertForQuestionAnswering = (
    transformers.AlbertForQuestionAnswering.from_pretrained("./model")
)
tokenizer: transformers.AlbertTokenizer = (
    transformers.AlbertTokenizerFast.from_pretrained("./model")
)

n_best_size = 20
max_answer_length = 30
null_score_diff_threshold = 0.0
max_length = 384
stride = 128

article = """Check out the project repository at [GitHub](https://github.com/vdyma/eureqa).

Training logs can be found at [ClearML](https://app.clear.ml/projects/cd2f4008afa34a68bd085588fe8f44e1/experiments/a971b54e499a4dbe8b90faf9b6969608/output/execution) (you must be registered to access the log).
"""

example_context = 'The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse ("Norman" comes from "Norseman") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries.'
example_question = "In what country is Normandy located?"


def predict(context: str, question: str) -> str:
    model_inputs = tokenizer(
        question,
        context,
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors="pt",
    )
    offset_mapping = model_inputs.pop("offset_mapping")
    model_inputs.pop("overflow_to_sample_mapping")
    output = model(**model_inputs)
    prelim_predictions = []
    min_null_prediction = None
    null_score = 0.0

    for i, offest in enumerate(offset_mapping):
        start_logits = output.start_logits[i]
        end_logits = output.end_logits[i]

        # Update minimum null prediction.
        feature_null_score = start_logits[0] + end_logits[0]
        min_null_prediction = {
            "offsets": (0, 0),
            "score": feature_null_score,
            "start_logit": start_logits[0],
            "end_logit": end_logits[0],
        }

        # Go through all possibilities for the `n_best_size` greater start and end logits.
        start_indexes = torch.argsort(start_logits, descending=True)[
            :n_best_size
        ].tolist()
        end_indexes = torch.argsort(end_logits, descending=True)[:n_best_size].tolist()
        for start_index in start_indexes:
            for end_index in end_indexes:
                # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                # to part of the input_ids that are not in the context.
                if (
                    start_index >= len(offest)
                    or end_index >= len(offest)
                    or offest[start_index] is None
                    or len(offest[start_index]) < 2
                    or offest[end_index] is None
                    or len(offest[end_index]) < 2
                ):
                    continue
                # Don't consider answers with a length that is either < 0 or > max_answer_length.
                if (
                    end_index < start_index
                    or end_index - start_index + 1 > max_answer_length
                ):
                    continue

                prelim_predictions.append(
                    {
                        "offsets": (
                            offest[start_index][0],
                            offest[end_index][1],
                        ),
                        "score": start_logits[start_index] + end_logits[end_index],
                        "start_logit": start_logits[start_index],
                        "end_logit": end_logits[end_index],
                    }
                )
    if min_null_prediction is not None:
        prelim_predictions.append(min_null_prediction)
        null_score = min_null_prediction["score"]

    # Only keep the best `n_best_size` predictions.
    predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[
        :n_best_size
    ]

    if min_null_prediction is not None and not any(
        p["offsets"] == (0, 0) for p in predictions
    ):
        predictions.append(min_null_prediction)

    # Use the offsets to gather the answer text in the original context.
    for pred in predictions:
        offsets = pred.pop("offsets")
        pred["text"] = context[offsets[0] : offsets[1]]

    # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
    # failure.
    if len(predictions) == 0 or (
        len(predictions) == 1 and predictions[0]["text"] == ""
    ):
        predictions.insert(
            0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0}
        )

    probs = torch.nn.functional.softmax(
        torch.tensor([pred.pop("score") for pred in predictions]), dim=0
    )

    # Include the probabilities in our predictions.
    for prob, pred in zip(probs, predictions):
        pred["probability"] = prob

    # Pick the best prediction.
    i = 0
    while predictions[i]["text"] == "":
        i += 1
    best_non_null_pred = predictions[i]

    # Then we compare to the null prediction using the threshold.
    score_diff = (
        null_score - best_non_null_pred["start_logit"] - best_non_null_pred["end_logit"]
    )
    return (
        "The context doesn't contain answer to this question."
        if score_diff > null_score_diff_threshold
        else best_non_null_pred["text"]
    )


if __name__ == "__main__":
    demo = gr.Interface(
        fn=predict,
        inputs=[
            gr.TextArea(label="Context", value=example_context),
            gr.Textbox(label="Question", value=example_question),
        ],
        outputs=gr.Textbox(label="Answer"),
        title="EureQA: Extractive Question Answering model",
        description="""EureQA is an extractive question answering model based on the [ALBERT Base v2](https://huggingface.co/albert/albert-base-v2) 
architecture and finetuned on [SQuAD 2.0](https://huggingface.co/datasets/rajpurkar/squad_v2) dataset.""",
        article=article,
    )
    demo.launch(share=True)
