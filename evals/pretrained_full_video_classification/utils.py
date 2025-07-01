import logging
from typing import Any


def rank_conditional_calls_wrapper(o, rank: int, whitelist_ranks: list[int] = [0]):

    class IgnoreCalls:

        def __getattribute__(self, name: str) -> Any:
            return lambda *args, **kwargs: None

    if rank in whitelist_ranks:
        return o
    else:
        return IgnoreCalls()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = rank_conditional_calls_wrapper(logging.getLogger(), rank=0)

    logger.info(f'I am rank 0.')
    logger = rank_conditional_calls_wrapper(logger, rank=1)
    logger.info('I should not be shown')