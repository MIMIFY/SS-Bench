import json
def check_and_parse_json(response):
    try:
        # 尝试解析为 JSON 格式
        response_dict = json.loads(response)
        # 检查是否包含必要的键
        if 'chapter' in response_dict and 'explanation' in response_dict:
            return response_dict
        else:
            return None
    except json.JSONDecodeError:
        # 如果无法解析为 JSON 格式，返回 None
        return None

def get_cur_chapter_stories(data_list):
	# 存储所有的章节名称
    chapters = set()

    # 存储章节与对应的标题
    chapter_title_map = {} # 字典
    chapter_story_map = {} # 字典
    chapter_story_list = [] # 字典的列表
    chapter_title_list = [] # 字典的列表
    # 遍历每个字典
    num_id = 0
    for item in data_list:
        chapter = item.get('Chapter')
        title = item.get('Title')
        story = {}
        story['id'] = "seed_story_" + str (num_id)
        num_id += 1
        story['chapter'] = chapter
        story['explanation'] = ""
        story['title'] = item.get('Title')
        story['introduction'] = item.get('Introduction')
        story['main_body'] = item.get('Main body')
        story['conclusion'] = item.get('Conclusion')
        # 将整个故事存储起来，[{'chapter':__,'explanation':__,....}]
        chapter_story_list.append(story)

        # 添加章节到集合中
        chapters.add(chapter)

        # 将故事内容添加到对应的章节字典里，{'chapter_name': list_of_story_dict}
        if chapter in chapter_story_map:
            chapter_story_map[chapter].append(story)
        else:
            chapter_story_map[chapter] = [story]

        # 将故事标题添加到对应的章节列表中 {'chapter_name':list_of_title}
        if chapter in chapter_title_map:
            chapter_title_map[chapter].append(title)
        else:
            chapter_title_map[chapter] = [title]

    # 输出所有的章节名称
    # print("所有的章节名称：", chapters, len(chapter))
    # 符合prompt格式的chapter_title_dict
    
    for chapter, titles in chapter_title_map.items():
        tmp = {}
        tmp['Chapter'] = chapter
        tmp['Social Story Titles in the chapter'] = titles
        chapter_title_list.append(tmp)

    # 输出章节与标题的对应关系
    # print("章节与标题的对应关系：", chapter_title_map)
    # 输出每个章节对应的故事标题列表
    # for chapter, storys in chapter_story_map.items():
    #     print(f"Chapter {chapter}:")
    #     print(len(storys))
    #     for story in storys:
    #         print(f"- {story['title']}")
    return chapters, chapter_title_list, chapter_story_list, chapter_story_map