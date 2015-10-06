#include <fstream>
#include <string>

using namespace std;

const std::string pattern = "	const unsigned int iBuild = ";

int main()
{
	fstream verFileOld, verFileNew;
	remove("Version.h_old");
	rename("Version.h", "Version.h_old");
	verFileOld.open("Version.h_old", ios::in);
	verFileNew.open("Version.h", ios::out);
	if (verFileNew.is_open() && verFileOld.is_open())
	{
		std::string line;
		while (std::getline(verFileOld, line))
		{
			if (!line.compare(0, pattern.size(), pattern))
			{
				std::string strBuild = line.substr(pattern.size());
				strBuild = strBuild.substr(0, strBuild.size() - 1);
				unsigned int iBuild = atoi(strBuild.c_str());
				++iBuild;
				verFileNew << pattern + std::to_string(iBuild) + ";\n";
			}
			else
			{
				verFileNew << line << endl;
			}
		}

		verFileNew.close();
		verFileOld.close();
	}
	else
	{
		return 1;
	}

	return 0;
}