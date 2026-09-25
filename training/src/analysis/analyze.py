import json

from collections import defaultdict


def analyze_trajectory():
    enterprise_trajectory = json.load(open("./enterprise_arena_gold.json", "r"))
    trajectory_num = len(enterprise_trajectory)

    tool_dict = defaultdict(int)
    number_of_steps = defaultdict(int)
    number_of_tools = defaultdict(int)
    number_of_errors = 0

    for trajectory in enterprise_trajectory:
        messages = trajectory["messages"]
        steps = 0
        tools = 0
        for message in messages:
            if message["role"] == "tool":
                tool_dict[message["name"]] += 1
                tools += 1
                if message["content"].startswith("Error:"):
                    number_of_errors += 1
            if message["role"] == "assistant":
                steps += 1
        number_of_steps[steps] += 1
        number_of_tools[tools] += 1

    print(f"Total trajectory number: {trajectory_num}")
    print("Tool usage:")
    for tool, count in tool_dict.items():
        print(f"  {tool}: {count}")
    print(number_of_steps)
    print(number_of_tools)
    print(number_of_errors)


def analyze_world_model_evaluation():
    #evaluation_result = json.load(open("checkpoints/elab_tool_output/checkpoint-1670/evaluation_metrics_test_wwm.json", "r"))
    evaluation_result = json.load(open("checkpoints/earena_tool_output/checkpoint-740/evaluation_metrics_test_wwm_gpt-4o-mini.json", "r"))
    baseline_results = evaluation_result["agent_replay_eval"]["baseline_actual_mcp"]
    assisted_results = evaluation_result["agent_replay_eval"]["world_model_assisted_actual_mcp"]
    imagined_results = evaluation_result["agent_replay_eval"]["imagined_trajectory_actual_mcp"]

    baseline_results = baseline_results["task_records"]
    assisted_results = assisted_results["task_records"]
    imagined_results = imagined_results["task_records"]

    baseline_calls = []
    assisted_calls = []
    imagined_calls = []

    baseline_completes = []
    assisted_completes = []
    imagined_completes = []

    baseline_f1 = []
    assisted_f1 = []
    imagined_f1 = []

    for i, (baseline_result, assisted_result, imagined_result) in enumerate(zip(baseline_results, assisted_results, imagined_results)):
        # if not baseline_result["trajectory_index"] in test_indices: continue
        if baseline_result["failure_reason"].startswith("unparseable"): continue
        if assisted_result["failure_reason"].startswith("unparseable"): continue
        if imagined_result["failure_reason"].startswith("unparseable"): continue

        if baseline_result["failure_reason"].startswith("final_answer_below"): baseline_result["completed"] = True
        if assisted_result["failure_reason"].startswith("final_answer_below"): assisted_result["completed"] = True
        if imagined_result["failure_reason"].startswith("final_answer_below"): imagined_result["completed"] = True
        
        # if baseline_result["tool_calls_taken"] == 0: continue
        # if assisted_result["tool_calls_taken"] == 0: continue
        # if imagined_result["tool_calls_taken"] == 0: continue
        # if assisted_result["tool_calls_taken"] == 15: continue

        # if baseline_result["final_answer_score"] is None: continue
        # if assisted_result["final_answer_score"] is None: continue
        # if imagined_result["final_answer_score"] is None: continue
        #if baseline_result["tool_calls_taken"] - assisted_result["tool_calls_taken"] > 1: print(i) 

        baseline_calls.append(baseline_result["tool_calls_taken"])
        assisted_calls.append(assisted_result["tool_calls_taken"])
        imagined_calls.append(imagined_result["tool_calls_taken"])

        baseline_completes.append(baseline_result["completed"])
        assisted_completes.append(assisted_result["completed"])
        imagined_completes.append(imagined_result["completed"])

        baseline_f1.append(baseline_result["final_answer_score"])
        assisted_f1.append(assisted_result["final_answer_score"])
        imagined_f1.append(imagined_result["final_answer_score"])

    print(baseline_calls)
    print(assisted_calls)
    print(imagined_calls)
    print(baseline_completes)
    print(assisted_completes)
    print(imagined_completes)
    print(baseline_f1)
    print(assisted_f1)
    print(imagined_f1)

    print(f"Count of test trajectories: {len(assisted_f1)}")
    print(f"Baseline - Average tool calls: {sum(baseline_calls)/len(baseline_calls)}, Completion rate: {sum(baseline_completes)/len(baseline_completes)}, Average F1: {sum(baseline_f1)/len(baseline_f1)}")
    print(f"Assisted - Average tool calls: {sum(assisted_calls)/len(assisted_calls)}, Completion rate: {sum(assisted_completes)/len(assisted_completes)}, Average F1: {sum(assisted_f1)/len(assisted_f1)}")
    print(f"Imagined - Average tool calls: {sum(imagined_calls)/len(imagined_calls)}, Completion rate: {sum(imagined_completes)/len(imagined_completes)}, Average F1: {sum(imagined_f1)/len(imagined_f1)}")


if __name__ == "__main__":
    #analyze_trajectory()
    analyze_world_model_evaluation()