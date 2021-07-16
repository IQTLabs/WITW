def get_aws_secrets(secrets_file):
    try:
        with open('./aws-secrets.txt', 'r') as secret_file:
            lines = secret_file.readlines()
            aki = lines[1].strip().split('=')[1]
            sak = lines[2].strip().split('=')[1]
            st = lines[3].strip().split('=')[1]
            return aki, sak, st
    except IOError as err:
        print(f'{err}')
        return None

def get_secret(secret_root, secret_name):
    try:
        with open(f'{secret_root}/{secret_name}', 'r') as secret_file:
            return secret_file.read()
    except IOError:
        return None