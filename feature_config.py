# coding:utf-8
#
# Copyright 2019 Tencent Inc.
# Author: Yang Xiao(mlhustxiao@tecent.com)
#

DENSE_FEATURES = ['userid_read_comment_sum', 'userid_read_comment_mean',
                  'userid_like_sum', 'userid_like_mean', 'userid_click_avatar_sum',
                  'userid_click_avatar_mean', 'userid_forward_sum', 'userid_forward_mean',
                  'userid_comment_sum', 'userid_comment_mean', 'userid_follow_sum',
                  'userid_follow_mean', 'userid_favorite_sum', 'userid_favorite_mean',
                  'feedid_read_comment_sum', 'feedid_read_comment_mean',
                  'feedid_like_sum', 'feedid_like_mean', 'feedid_click_avatar_sum',
                  'feedid_click_avatar_mean', 'feedid_forward_sum', 'feedid_forward_mean',
                  'feedid_comment_sum', 'feedid_comment_mean', 'feedid_follow_sum',
                  'feedid_follow_mean', 'feedid_favorite_sum', 'feedid_favorite_mean',
                  'authorid_read_comment_sum', 'authorid_read_comment_mean',
                  'authorid_like_sum', 'authorid_like_mean', 'authorid_click_avatar_sum',
                  'authorid_click_avatar_mean', 'authorid_forward_sum',
                  'authorid_forward_mean', 'authorid_comment_sum',
                  'authorid_comment_mean', 'authorid_follow_sum', 'authorid_follow_mean',
                  'authorid_favorite_sum', 'authorid_favorite_mean', 'videoplayseconds']
SINGLE_ID_FEATURES = ['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id']
MULTI_ID_FEATURES = ['description', 'ocr', 'asr', 'manual_keyword_list',
                     'manual_tag_list', 'description_char', 'ocr_char', 'asr_char',
                     'user_read_comment_feedid_sequence', 'user_read_comment_authorid_sequence',
                     'user_like_feedid_sequence', 'user_like_authorid_sequence',
                     'user_click_avatar_feedid_sequence', 'user_click_avatar_authorid_sequence',
                     'user_forward_feedid_sequence', 'user_forward_authorid_sequence',
                     'user_comment_feedid_sequence', 'user_comment_authorid_sequence',
                     'user_follow_feedid_sequence', 'user_follow_authorid_sequence',
                     'user_favorite_feedid_sequence', 'user_favorite_authorid_sequence',
                     'user_has_action_feedid_sequence', 'user_has_action_authorid_sequence']
FEATURE_MAP_DIC = {
    'user_read_comment_feedid_sequence': 'feedid',
    'user_like_feedid_sequence': 'feedid',
    'user_click_avatar_feedid_sequence': 'feedid',
    'user_forward_feedid_sequence': 'feedid',
    'user_comment_feedid_sequence': 'feedid',
    'user_follow_feedid_sequence': 'feedid',
    'user_favorite_feedid_sequence': 'feedid',
    'user_has_action_feedid_sequence': 'feedid',
    'user_read_comment_authorid_sequence': 'authorid',
    'user_like_authorid_sequence': 'authorid',
    'user_click_avatar_authorid_sequence': 'authorid',
    'user_forward_authorid_sequence': 'authorid',
    'user_comment_authorid_sequence': 'authorid',
    'user_follow_authorid_sequence': 'authorid',
    'user_favorite_authorid_sequence': 'authorid',
    'user_has_action_authorid_sequence': 'authorid'
}
