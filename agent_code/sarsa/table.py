import json


class Table(dict):
    def __missing__(self, key):
        self[key] = [0.0] * 6
        return self[key]

    def save_to_json(self, file_path='q_table.json'):
        with open(file_path, 'w') as file:
            json.dump(self, file, indent=4)
    
    def load_from_json(self, file_path='q_table.json'):
        with open(file_path, 'r') as file:
            data = json.load(file)
            self.update(data)