from lxml import html
import requests
import mysql.connector


def get_name(drug):
    page = requests.get('http://www.kegg.jp/kegg-bin/search?q='+drug+'&display=drug&compound_thumbnail=&pathway_thumbnail=&drug_thumbnail=&from=drug&lang=&target=compound%2Bdrug%2Bdgroup%2Benviron%2Bdisease')
    tree = html.fromstring(page.content)

    buyers = tree.xpath('//td[@class="data1"]/text()')
    buyers = buyers[2] #(''.join(buyers)).lstrip()
    buyers = str(buyers).strip()
    # print(buyers)
    return buyers

if __name__ == '__main__':
    cnx = mysql.connector.connect(user='root', password='123',
                                  host='127.0.0.1',
                                  database='thesisDataset')
    cursor = cnx.cursor()

    query = ("select a,b from all_drugs")
    x = cursor.execute(query)


    # query = ("select a,b from all_drugs")
    # x = cursor2.execute(query)

    for i in cursor:
        cnx2 = mysql.connector.connect(user='root', password='123',
                                       host='127.0.0.1',
                                       database='thesisDataset')
        cursor2 = cnx2.cursor()

        print(i[0] , get_name(i[0]))
        string = "INSERT INTO drug_with_names (a, b, c) VALUES ('"+str(i[0])+"', '"+str(i[1])+"', '"+str(get_name(i[0]))+"');"
        # string = "INSERT INTO drug_with_names (a, b, c) VALUES ('6', 'h', 'h');"
        # print(string)
        query = string
        try:
            x = cursor2.execute(query)
            cnx2.commit()

            cnx2.close()
            # break
        except:
            print('skipped')
    cnx.close()