from kedro.pipeline import Pipeline, node, pipeline

from .nodes import  split_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func= split_data,
                inputs= ["ingested_data","parameters"],
                outputs=["ref_data","ana_data"],
                name="split_out_of_sample",
            ),
        ]
    )
