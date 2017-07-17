#ifndef CDF_BSEARCH_H
#define CDF_BSEARCH_H

#include <vector>
#include <valarray>

template<class T>
unsigned int cdf_bsearch(T xi, const std::vector<T>& cdf_table)
{
  unsigned int table_size = cdf_table.size();
  unsigned int middle = table_size = table_size>>1;
  unsigned int odd = 0;
  while(table_size > 0)
  {
    odd = table_size&1;
    table_size = table_size>>1;
    if(xi > cdf_table[middle])
      middle += table_size + odd;
    else if(xi < cdf_table[middle - 1])
      middle -= table_size + odd;
    else
      break;
  }
  return middle;
}

template<class T>
unsigned int cdf_bsearch(T xi, const std::valarray<T>& cdf_table)
{
  unsigned int table_size = cdf_table.size();
  unsigned int middle = table_size = table_size>>1;
  unsigned int odd = 0;
  while(table_size > 0)
  {
    odd = table_size&1;
    table_size = table_size>>1;
    if(xi > cdf_table[middle])
      middle += table_size + odd;
    else if(xi < cdf_table[middle - 1])
      middle -= table_size + odd;
    else
      break;
  }
  return middle;
}

template<class T>
unsigned int cdf_bsearch(T xi, T* cdf_table, unsigned int table_size)
{
  unsigned int middle = table_size = table_size>>1;
  unsigned int odd = 0;
  while(table_size > 0)
  {
    odd = table_size&1;
    table_size = table_size>>1;
    if(xi > cdf_table[middle])
      middle += table_size + odd;
    else if(xi < cdf_table[middle - 1])
      middle -= table_size + odd;
    else
      break;
  }
  return middle;
}

#endif // CDF_BSEARCH_H