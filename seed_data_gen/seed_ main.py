import json
from openai_access import call_chatgpt
from utils import get_cur_chapter_stories,check_and_parse_json
from explain_chapter import createExplainChapterPrompt



# 获取原始章节的解释列表，为explan-then-generate做准备
def get_original_chapter_explanation_list(all_objs):
    chapter_explanation = []
    chapter_explanation_map_dict = {}
    chapters, chapter_title_list, chapter_story_list, _ = get_cur_chapter_stories(all_objs)
    for i in chapter_title_list:
        prompt = createExplainChapterPrompt(i)
        print(prompt)
        
        answer = call_chatgpt(prompt)
        print("-----",answer)
        tmp = {}
        # 注意根据需求，把key调整大小写
        tmp['Chapter'] = i['Chapter']
        tmp['Explanation'] = answer
        chapter_explanation_map_dict[i['Chapter']] = answer
        chapter_explanation.append(tmp)
    
    # 返回一个列表，每个元素为一个字典，key分别为 Chapter 和 Explanation
    return chapter_explanation


        
# 下面获取种子池
def construct_seed_stories_seed_titles(ori_jsonfile, ori_explanation_jsonfile, seed_chapter_stories_filename, seed_chapter_titles_filename):

    # 打开JSON文件并读取最原始的数据
    with open(ori_jsonfile, 'r', encoding='utf-8') as fr:
        all_objs = json.load(fr)
    chapters, chapter_title_list, chapter_story_list, _ = get_cur_chapter_stories(all_objs)

    # 构建种子故事集，在已有的chapter_story_list中添加，每个chapter对应的explanation描述
    with open(ori_explanation_jsonfile, 'r', encoding='utf-8') as fr:
        chapter_explanation_list = json.load(fr)
    # 获取chapter与explanation的匹配关系，key为chapter，值为explanation
    chapter_explanation_map_dict = {}
    for i in chapter_explanation_list:
        chapter_explanation_map_dict[i['Chapter']] = i['Explanation']
    
    # 获取种子故事集
    for story in chapter_story_list:
        story['explanation'] = chapter_explanation_map_dict[story['chapter']]
    
    with open(seed_chapter_stories_filename+'.json', 'w',encoding='utf-8') as f:#ensure_ascii=False
        json.dump(chapter_story_list, f, ensure_ascii=False, indent=4)
    with open(seed_chapter_stories_filename+'.jsonl', 'w', encoding='utf-8') as jsonl_file:
        # 遍历JSON数组中的每个元素
        for item in chapter_story_list:
            # 将每个元素转换为字符串并写入JSONL文件
            jsonl_file.write(json.dumps(item,ensure_ascii=False) + '\n')
    print("获取种子故事集，完毕！")

    # 获取种子标题集
    for title in chapter_title_list:
        title['Explanation'] = chapter_explanation_map_dict[title['Chapter']]
    
    with open(seed_chapter_titles_filename+'.json', 'w',encoding='utf-8') as f:#ensure_ascii=False
        json.dump(chapter_title_list, f, ensure_ascii=False, indent=4)
    with open(seed_chapter_titles_filename+'.jsonl', 'w', encoding='utf-8') as jsonl_file:
        # 遍历JSON数组中的每个元素
        for item in chapter_title_list:
            # 将每个元素转换为字符串并写入JSONL文件
            jsonl_file.write(json.dumps(item,ensure_ascii=False) + '\n')
    print("获取种子标题集，完毕！")
    
    return 0



def main():
    # 打开JSON文件并读取最原始的数据
    json_file = "github/seed_data_gen/seed_data/pure-ori.json"
    with open(json_file, 'r', encoding='utf-8') as fr:
        all_objs = json.load(fr)
        
    # zero-shot，根据chapter以及其所拥有的所有title，让gpt为此chapter生成一段文字解释，满足后续的explain-then-generate的要求
    chapter_explanation_list = get_original_chapter_explanation_list(all_objs)
    # 将获得的list_of_dicts写入json文件中保存
    with open('github/seed_data_gen/seed_data/seed_chapters_explanations.json', 'w') as f:	
        json.dump(chapter_explanation_list, f,ensure_ascii=False, indent=4)


    ori_jsonfile = 'github/seed_data_gen/seed_data/pure-ori.json'
    ori_explanation_jsonfile = 'github/seed_data_gen/seed_data/seed_chapters_explanations.json'
    seed_chapter_stories_filename = 'github/seed_data_gen/seed_data/seed_chapter_story_list'
    seed_chapter_titles_filename = 'github/seed_data_gen/seed_data/seed_chapter_title_list'
    construct_seed_stories_seed_titles(ori_jsonfile, ori_explanation_jsonfile, seed_chapter_stories_filename, seed_chapter_titles_filename)

    print("完毕")
        
    # 转换json,jsonl文件格式函数
    def transform_from_jsonfile_to_jsonlfile(json_file, jsonl_file):
        # 读json文件, json.load(文件)
        with open(json_file, 'r', encoding='utf-8') as fr:
            list_of_dicts = json.load(fr)
        # 写入jsonl文件，json.dumps(单行数据)
        with open(jsonl_file, 'w', encoding='utf-8') as jsonl_file:
            # 遍历JSON数组中的每个元素
            for item in list_of_dicts:
                # 将每个元素转换为字符串并写入JSONL文件
                jsonl_file.write(json.dumps(item,ensure_ascii=False) + '\n')
        print("json文件已转换为jsonl文件。")
        return 0

    # 下面是在转换json文件格式为jsonl文件格式
    json_file = 'github/seed_data_gen/seed_data/seed_chapters_explanations.json'
    jsonl_file = 'github/seed_data_gen/seed_data/seed_chapters_explanations.jsonl'
    transform_from_jsonfile_to_jsonlfile(json_file, jsonl_file)
    print("完毕")


if __name__ == "__main__": 
    main()

        




