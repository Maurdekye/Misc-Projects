using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Windows.Forms;

namespace CachedSearch
{
    public partial class Form1 : Form
    {
        CachedSearchManager searchManager;

        public Form1()
        {
            InitializeComponent();
            Assembly assembly = Assembly.GetExecutingAssembly();
            StreamReader sread = new StreamReader(assembly.GetManifestResourceStream("CachedSearch.Thesaurus.txt"));
            string[] contents = sread.ReadToEnd().Split(new char[] {'\n'});
            List<string> data = new List<string>(contents);
            searchManager = new CachedSearchManager(data);
        }

        private void searchBox_TextChanged(object sender, EventArgs e)
        {
            searchBox.Text = searchBox.Text.ToLower();
            string searchTerm = searchBox.Text.Trim();
            displayLabel.Text = "";
            if (searchTerm == "")
                return;
            List<string> searchResults = searchManager.Search(searchTerm);
            for (int i=0;i<45;i++)
            {
                if (i >= searchResults.Count)
                    break;
                displayLabel.Text += searchResults[i] + "\n";
            }
        }
    }
}

public class CachedSearchManager
{
    public List<string> Dataset;
    public List<CacheItem> Cacheset;

    public CachedSearchManager(List<string> dataset)
    {
        this.Dataset = dataset;
        this.Cacheset = new List<CacheItem>();
    }

    private static List<string> Filter(string term, List<string> data)
    {
        List<string> finalList = new List<string>();
        foreach (string item in data)
        {
            if (item.Contains(term))
                finalList.Add(item);
        }
        return finalList;
    }

    private CacheItem GetCacheItemForTerm(string term)
    {
        int LongestCachedTerm = -1;
        for (int i=0;i<Cacheset.Count;i++)
        {
            CacheItem item = Cacheset[i];
            if (term == item.Term)
                return item;
            else if (term.StartsWith(item.Term) &&
                (LongestCachedTerm == -1 || Cacheset[LongestCachedTerm].Term.Length < item.Term.Length))
                LongestCachedTerm = i;
        }
        if (LongestCachedTerm == -1)
            return null;
        else
            return Cacheset[LongestCachedTerm];
    }

    public List<string> Search(string term)
    {
        CacheItem fromCache = GetCacheItemForTerm(term);
        List<string> localResults;
        if (fromCache == null)
            localResults = CachedSearchManager.Filter(term, Dataset);
        else if (fromCache.Term == term)
            return fromCache.Results;
        else
            localResults = CachedSearchManager.Filter(term, fromCache.Results);
        Cacheset.Add(new CacheItem(term, localResults));
        return localResults;
    }

    public class CacheItem
    {
        public string Term;
        public List<string> Results;

        public CacheItem(string term, List<string> results)
        {
            this.Term = term;
            this.Results = results;
        }
    }
}