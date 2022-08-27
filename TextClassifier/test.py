
# if __name__ == '__main__':
#     with open("./THUCNews/data/train.txt", "a") as train:  # 打开测试文件
#         with open("./THUCNews/data/test.txt", "a") as test:  # 打开文件
#             with open("./THUCNews/data/dev.txt", "a") as dev:  # 打开文件
#                 with open("./THUCNews/data/all_data.txt", "r") as all_data:  # 打开文件
#                     temp = 0
#                     while True:
#                         data1 = all_data.readline()  # 读取文件
#                         data = data1[:-1]
#                         if len(data) == 0:
#                             break
#                         num = data[-1]
#                         if data:
#                             if num == '9':
#                                 temp = temp + 1
#                                 if temp < 1250:
#                                     train.writelines(data1)
#                                 elif temp < 1500:
#                                     test.writelines(data1)
#                                 elif temp < 1563:
#                                     dev.writelines(data1)
#                                 else:
#                                     break
#                 all_data.close()
#             dev.close()
#         test.close()
#     train.close()

# if __name__ == '__main__':
#     with open("./THUCNews/data/all_data1.txt", "a") as all_data1:  # 打开测试文件
#         with open("./THUCNews/data/all_data.txt", "r") as all_data:  # 打开文件
#             while True:
#                 data1 = all_data.readline()  # 读取文件
#                 data = data1[:-1]
#                 if len(data) == 0:
#                     break
#                 num = data[-1]
#                 subdata = data[:20] + '  ' + num
#                 if data:
#                     all_data1.writelines(subdata)
#                 else:
#                     break
#         all_data.close()
#     all_data1.close()

# if __name__ == '__main__':
#     with open("./THUCNews/data/train1.txt", "a") as train1:  # 打开测试文件
#         with open("./THUCNews/data/train.txt", "r") as train:  # 打开文件
#             while True:
#                 data1 = train.readline()  # 读取文件
#                 data = data1[:-1]
#                 if len(data) == 0:
#                     break
#                 num = data[-1]
#                 subdata = data[:20] + '  ' + num + '\n'
#                 if data:
#                     train1.writelines(subdata)
#                 else:
#                     break
#         train.close()
#     train1.close()

# if __name__ == '__main__':
#     with open("./THUCNews/data/test1.txt", "a") as test1:  # 打开测试文件
#         with open("./THUCNews/data/test.txt", "r") as test:  # 打开文件
#             while True:
#                 data1 = test.readline()  # 读取文件
#                 data = data1[:-1]
#                 if len(data) == 0:
#                     break
#                 num = data[-1]
#                 subdata = data[:20] + '  ' + num + '\n'
#                 if data:
#                     test1.writelines(subdata)
#                 else:
#                     break
#         test.close()
#     test1.close()

# if __name__ == '__main__':
#     with open("./THUCNews/data/dev1.txt", "a") as dev1:  # 打开测试文件
#         with open("./THUCNews/data/dev.txt", "r") as dev:  # 打开文件
#             while True:
#                 data1 = dev.readline()  # 读取文件
#                 data = data1[:-1]
#                 if len(data) == 0:
#                     break
#                 num = data[-1]
#                 subdata = data[:20] + '  ' + num + '\n'
#                 if data:
#                     dev1.writelines(subdata)
#                 else:
#                     break
#         dev.close()
#     dev1.close()

# if __name__ == '__main__':
#     with open("./THUCNews/data/all_data.txt", "r") as all_data:  # 打开测试文件
#         num0 = 0
#         num1 = 0
#         num2 = 0
#         num3 = 0
#         num4 = 0
#         num5 = 0
#         num6 = 0
#         num7 = 0
#         num8 = 0
#         num9 = 0
#         while True:
#             data1 = all_data.readline()  # 读取文件
#             data = data1[:-1]
#             if len(data) == 0:
#                 break
#             num = data[-1]
#             if data:
#                 if num == '0':
#                     num0 = num0 +1
#                 if num == '0':
#                     num1 = num1 +1
#                 if num == '0':
#                     num2 = num2 +1
#                 if num == '0':
#                     num3 = num3 +1
#                 if num == '0':
#                     num4 = num4 +1
#                 if num == '0':
#                     num5 = num5 +1
#                 if num == '0':
#                     num6 = num6 +1
#                 if num == '0':
#                     num7 = num7 +1
#                 if num == '0':
#                     num8 = num8 +1
#                 if num == '0':
#                     num9 = num9 +1
#             else:
#                 break
#         print('num0:', num0, '\n')
#         print('num1:', num1, '\n')
#         print('num2:', num2, '\n')
#         print('num3:', num3, '\n')
#         print('num4:', num4, '\n')
#         print('num5:', num5, '\n')
#         print('num6:', num6, '\n')
#         print('num7:', num7, '\n')
#         print('num8:', num8, '\n')
#         print('num9:', num9, '\n')
#     all_data.close()

if __name__ == '__main__':
    a = 0.9257 + 0.9328 + 0.9823 + 0.9554 + 0.9581 + 0.9184 + 0.9663 + 0.9387
    print(a / 8)