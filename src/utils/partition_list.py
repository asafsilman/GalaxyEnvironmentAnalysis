def partition_list(list_to_partition: list, percentage: float):
    split_index = round(len(list_to_partition) * percentage)
    return list_to_partition[:split_index], list_to_partition[split_index:]