import unittest
from music_rag import MusicRetriever,load_catalog,validate_generation,normalize_query

class MusicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog=load_catalog();cls.retriever=MusicRetriever(cls.catalog)
    def test_real_catalog(self):
        self.assertGreater(len(self.catalog),5000)
        self.assertEqual(len(self.catalog),len({r['id'] for r in self.catalog}))
    def test_retrieval(self):
        hits=self.retriever.search('relaxing acoustic guitar',k=3,required_terms=['guitar'])
        self.assertEqual(len(hits),3)
        self.assertTrue(all('guitar' in (r['caption']+' '.join(r['aspects'])).lower() for r in hits))
        self.assertEqual([r['score'] for r in hits],sorted([r['score'] for r in hits],reverse=True))
    def test_unknown_empty(self):
        self.assertEqual(self.retriever.search('zzxqvunknownword'),[])
        with self.assertRaises(ValueError): self.retriever.search(' ')
        with self.assertRaises(ValueError): self.retriever.search('jazz',k=0)
    def test_exclusion(self):
        key=self.catalog[0]['id']
        self.assertNotIn(key,[r['id'] for r in self.retriever.similar(key)])
        self.assertEqual(self.retriever.search('piano',required_terms=['zzxqvunknownword']),[])
    def test_french(self):
        self.assertIn('guitar',normalize_query('guitare acoustique douce'))
        self.assertEqual(normalize_query('sans paroles'),'instrumental')
    def test_grounding(self):
        row=self.catalog[0]
        good={'recommendations':[{'id':row['id'],'evidence':row['caption']}]}
        self.assertEqual(validate_generation(good,[row])['recommendations'][0]['url'],row['url'])
        for bad in [{'recommendations':[{'id':'invented','evidence':'x'}]},
                    {'recommendations':[{'id':row['id'],'evidence':'invented song title'}]},
                    {'recommendations':good['recommendations']*2}]:
            with self.assertRaises(ValueError): validate_generation(bad,[row])
if __name__=='__main__': unittest.main()
