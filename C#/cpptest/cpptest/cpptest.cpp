// cpptest.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <ctime>
#include <chrono>
#include <vector>
#include <sstream>
#include <tuple>
using namespace std;

long long int mtime()
{
	return chrono::duration_cast <chrono::milliseconds> (chrono::system_clock::now().time_since_epoch()).count();
}

void rnd(long long int maxrnd)
{
	long long int mstart = mtime();
	long long int cnt = 0;
	long long int n = rand() % maxrnd;
	while (rand() % maxrnd != n)
	{
		cnt++;
	}
	long long int mend = mtime();
	long long int mdiff = mend - mstart;
	cout << "num is " << cnt << ", took " << mdiff << " ms" << endl;
}


void ary(int num)
{
	long long int stime = mtime();
	long long int ary[10000] = {};
	for (int i = 0; i < 10000; i++)
	{
		ary[i] = 0;
	}
	for (int n = 0; n < num; n++)
	{
		for (int i = 0; i < 10000; i++)
		{
			ary[i]++;
		}
	}
	long long int etime = mtime();
	cout << "took " << (etime - stime) << " ms" << endl;
}

string ptbl(vector<vector<int>> tbl)
{
	stringstream pbuff;
	for (int y = 0; y < tbl.size(); y++)
	{
		for (int x = 0; x < tbl.at(0).size(); x++)
		{
			int val = tbl.at(y).at(x);
			if (val == 0)
			{
				pbuff << ". ";
			}
			else
			{
				pbuff << val << " ";
			}
		}
		pbuff << endl;
	}
	pbuff << endl;
	return pbuff.str();
}


vector<vector<int>> inittbl(int w, int h, int v)
{
	vector<vector<int>> tbl;
	for (int y = 0; y < h; y++)
	{
		vector<int> intbl = vector<int>();
		for (int x = 0; x < w; x++)
		{
			intbl.push_back(v);
		}
		tbl.push_back(intbl);
	}
	return tbl;
}

void settbl(vector<vector<int>> * tbl, int x, int y, int val)
{
	tbl->at(y).at(x) = val;
}

int gettbl(vector<vector<int>> * tbl, int x, int y)
{
	return tbl->at(y).at(x);
}

void fld(vector<vector<int>> * tbl, int x, int y, int val)
{
	if (y < 0 || y >= tbl->size() || x < 0 || x >= tbl->at(0).size())
	{
		return;
	}
	if (gettbl(tbl, x, y) != 0)
	{
		return;
	}
	settbl(tbl, x, y, val);
	int dr[2] = { -1, 1 };
	int dt[2] = { 1, 0 };
	for (int o = 0; o < 2; o++)
	{
		for (int d = 0; d < 2; d++)
		{
			fld(tbl, x + dr[o] * d, y + dr[o] * dt[d], val);
		}
	}
}
void lpfld(vector<vector<int>> * tbl, int xin, int yin, int val)
{
	int dr[2] = { -1, 1 };
	int dt[2] = { 1, 0 };
	vector<tuple<int, int>> * posl = new vector<tuple<int, int>>();
	posl->push_back(make_tuple(xin, yin));
	while (posl->size() > 0)
	{
		vector<tuple<int, int>> * nposl = new vector<tuple<int, int>>();
		for (int i = 0; i < posl->size(); i++)
		{
			int x = get<0>(posl->at(i));
			int y = get<1>(posl->at(i));
			if (y < 0 || y >= tbl->size() || x < 0 || x >= tbl->at(0).size())
			{
				continue;
			}
			if (gettbl(tbl, x, y) != 0)
			{
				continue;
			}
			settbl(tbl, x, y, val);
			for (int o = 0; o < 2; o++)
			{
				for (int d = 0; d < 2; d++)
				{
					int xp = x + dr[o] * d;
					int yp = y + dr[o] * dt[d];
					nposl->push_back(make_tuple(xp, yp));
				}
			}
		}
		posl = nposl;
	}
}

int main()
{
	vector<vector<int>> tbl = inittbl(70, 70, 0);
	for (int i = 10; i <= 30; i++)
	{
		settbl(&tbl, 10, i, 1);
	}
	for (int i = 10; i <= 30; i++)
	{
		settbl(&tbl, i, 10, 1);
	}
	for (int i = 10; i <= 20; i++)
	{
		settbl(&tbl, 30, i, 1);
	}
	for (int i = 10; i <= 20; i++)
	{
		settbl(&tbl, i, 30, 1);
	}
	for (int i = 30; i <= 50; i++)
	{
		settbl(&tbl, i, 20, 1);
	}
	for (int i = 30; i <= 50; i++)
	{
		settbl(&tbl, 20, i, 1);
	}
	for (int i = 20; i <= 50; i++)
	{
		settbl(&tbl, i, 50, 1);
	}
	for (int i = 20; i <= 50; i++)
	{
		settbl(&tbl, 50, i, 1);
	}
	cout << ptbl(tbl);
	//system("PAUSE");
	lpfld(&tbl, 5, 5, 2);
	cout << ptbl(tbl);
	system("PAUSE");

	/*cout << "the cats and the cradle and the silver spoon" << endl;
	cout << "little boy blue and the man on the moon" << endl;
	cout << "when you coming home son i don't know when" << endl;
	cout << "but we'll get together then, yeah" << endl;
	cout << "you know we'll have a good time then" << endl;*/

}