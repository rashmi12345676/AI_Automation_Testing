def log_decorator(func):
    def wrapper(*args , **kwargs):
        print(f"[LOG] Calling {func.__name__} with arguments: {args}")
        print(f"[LOG] Calling {func.__name__} with arguments: {kwargs}")
        return func(*args, **kwargs)
    return wrapper

@log_decorator
def transfer_funds(amount, account_id):
    print(f"Transferred {amount} to {account_id}")


transfer_funds(500, "ACC123")
transfer_funds(1000, [1,2,3,4,3])


def audit_log(func):
    def wrapper():
        print("Attempting to run: [function_name]")
        return func
    print("Finished [function_name] in [X] seconds")
    return (wrapper)

@audit_log
def call():
    return "Hi i am function baby "
    
