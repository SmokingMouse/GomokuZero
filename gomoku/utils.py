import time

def timer(func):
    """一个简单的装饰器，用于打印函数的执行时间。"""
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)  # 执行原始函数
        end_time = time.time()    # 记录结束时间
        
        elapsed_time = end_time - start_time
        print(f"函数 '{func.__name__}' 执行耗时: {elapsed_time:.4f} 秒")
        
        return result
    return wrapper