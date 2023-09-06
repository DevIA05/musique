import psycopg2
import os
    
    
def insert_genre(genre):
    try:
    
        conn = psycopg2.connect(
            database=os.environ['DBNAME'],
            user=os.environ['DBUSER'],
            password=os.environ['DBPWD'],
            host=os.environ['DBHOST'],
            port="5432"
        )
        
        cursor = conn.cursor()

        cursor.execute("INSERT INTO test (genre) VALUES (%s)", (genre,))
        
        conn.commit()
    except Exception as e:
        print("erreur",e)
    finally:
        cursor.close()
        conn.close()
