import torch
from typing import List

from models.graph_T5.autoregressive_GLM import GraphT5ForConditionalGeneration
from models.graph_T5.wrapper_functions import (
    Graph,
    graph_to_graphT5,
    get_embedding,
    Data,
    add_text_to_graph_data,
)
from seminars.kolber.GraphLanguageModels.models.graph_T5.graph_t5.tokenization_t5 import (
    T5Tokenizer,
)


def get_batch(data_instances: List[Data], pad_token_id: int, device: str):
    """
    slightly simplified version of the get_batch in experiments/encoder/train_LM.py
    """
    max_seq_len = max([data.input_ids.shape[1] for data in data_instances])

    # intialize tensors
    input_ids = (
        torch.ones((len(data_instances), max_seq_len), dtype=torch.long, device=device)
        * pad_token_id
    )
    relative_position = torch.zeros(
        (len(data_instances), max_seq_len, max_seq_len), dtype=torch.long, device=device
    )
    sparsity_mask = torch.zeros(
        (len(data_instances), max_seq_len, max_seq_len), dtype=torch.bool, device=device
    )
    use_additional_bucket = torch.zeros(
        (len(data_instances), max_seq_len, max_seq_len), dtype=torch.bool, device=device
    )

    # fill tensors
    for i, data in enumerate(data_instances):
        input_ids[i, : data.input_ids.shape[1]] = data.input_ids
        relative_position[
            i, : data.relative_position.shape[1], : data.relative_position.shape[2]
        ] = data.relative_position
        sparsity_mask[
            i, : data.sparsity_mask.shape[1], : data.sparsity_mask.shape[2]
        ] = data.sparsity_mask
        use_additional_bucket[
            i,
            : data.use_additional_bucket.shape[1],
            : data.use_additional_bucket.shape[2],
        ] = data.use_additional_bucket

    indices = [data.indices for data in data_instances]

    return input_ids, relative_position, sparsity_mask, use_additional_bucket, indices


def main():
    # define random parameters
    modelsize = "t5-small"
    init_additional_buckets_from = 1e6

    # define test inputs (2 instances to implement batching)
    graph1 = [
        ("dog", "is a", "animal"),
        ("cat", "is a", "animal"),
        ("black poodle", "is a", "dog"),
    ]
    graph2 = [
        ("subject1", "relation1", "object1"),
        ("subject2", "relation2", "object1"),
        ("subject3", "relation3", "subject1"),  # subject1 is the object of this triplet
    ]
    graphs = [Graph(graph1), Graph(graph2)]
    query_concepts = [
        "dog",
        "relation1",
    ]  # concepts or relation which is classified. Can be a relation aswell, as long as the reation occurs only once.
    # For instance, this can be used to predict masked relations.
    texts = [
        "The black poodle chases a cat.",
        "This is an example text for the second graph.",
    ]

    # 2 different classifiers (lGLM, gGLM with and without text)
    params = [
        {
            "name": "lGLM w/o text",
            "num_additional_buckets": 0,
            "how": "local",
            "use_text": False,
        },
        {
            "name": "gGLM w/o text",
            "num_additional_buckets": 1,  # gGLM needs one additional bucket for the global graph-to-graph relative position
            "how": "global",
            "use_text": False,
        },
    ]

    for param in params:

        # load model
        model = GraphT5ForConditionalGeneration(
            config=GraphT5ForConditionalGeneration.get_config(
                modelsize=modelsize,
                num_additional_buckets=param["num_additional_buckets"],
            )
        )
        tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")

        # initialize additional buckets. The additional buckets are the additional relative posistions introduced in gGLM and in text-guided models.
        if param["num_additional_buckets"] > 0:
            model.t5model.encoder.init_relative_position_bias(
                modelsize=modelsize,
                init_decoder=False,
                init_additional_buckets_from=init_additional_buckets_from,
            )

        print()  # loading the models gives warnings that nor all aprameters are used (because the decoder paramters are not used) and if additional buckets are used, then there also is a warning that they are initialized randomly. init_relative_position_bias also loads the model internally, so the warnings are printed twice.
        print(f"Model: {param['name']}")

        # preprocess data, i.e., convert graphs (and optionally the text) to relative position matrix, sparsity matrix, input ids, etc
        data = []
        for g, t in zip(graphs, texts):
            tmp_data = graph_to_graphT5(g, model.tokenizer, how=param["how"], eos=False)
            add_text_to_graph_data(
                data=tmp_data,
                text=t,
                tokenizer=model.tokenizer,
                use_text=param["use_text"],
            )
            data.append(tmp_data)

        # get batch
        input_ids, relative_position, sparsity_mask, use_additional_bucket, indices = (
            get_batch(data, model.tokenizer.pad_token_id, "cpu")
        )

        # generate and decode
        outputs = model.generate(
            input_ids=input_ids,
            relative_position=relative_position,
            sparsity_mask=sparsity_mask,
            use_additional_bucket=use_additional_bucket,
        )
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
