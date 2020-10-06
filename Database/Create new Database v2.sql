/* #################################      1      ###############################################
   ################################# NRGSearch DB ################################################ */

-- Create Database
CREATE DATABASE NRGSearch;

-- Run the script Initial_View_CrossDatabase.py from Database folder in order to construct the PROTUCTSOLD and FINANCE tables

-- For problem with Assembly
ALTER DATABASE NRG SET TRUSTWORTHY ON;
GO
USE NRG
GO
EXEC sp_changedbowner 'sa'
GO

-- ################################# Table PRODUCTSOLD ############################################### --
/*
SELECT *
INTO NRGSearch.dbo.PRODUCTSOLD
FROM NRG.dbo.INIT
*/

ALTER TABLE NRGSearch.dbo.PRODUCTSOLD
ALTER COLUMN ProductNo VARCHAR(40) NOT NULL

ALTER TABLE NRGSearch.dbo.PRODUCTSOLD
ADD	CONSTRAINT PK_NSProducts PRIMARY KEY (ProductNo);

-- Alter Table PRODUCTSOLD with new characteristics (Removed TextureID, FabricID, ShapeID, PartID, StyleID) 
ALTER TABLE NRGSearch.dbo.PRODUCTSOLD
ADD ClusterID INT NOT NULL DEFAULT -1,
	FClusterID INT NOT NULL DEFAULT -1

-- ################################# Create INTERMEDIATE Tables ############################################### --
-- 1) Table PRODUCTSUBCATEGORY
CREATE TABLE NRGSearch.dbo.PRODUCTSUBCATEGORY (
    ProductSubcategoryID INT,
	ProductSubcategory VARCHAR(100),
	CONSTRAINT PK_NProductSubcategory PRIMARY KEY (ProductSubcategoryID)

);
INSERT INTO NRGSearch.dbo.PRODUCTSUBCATEGORY (ProductSubcategoryID,ProductSubcategory)
VALUES 
(1,'SHORT SET'),(2,'SHORTS'),(3,'BLOUSE POLO SHORT SLEEVE'),(4,'JEANS'),(5,'INFANT ROMPER'),(6,'SHIRT'),(7,'JACKET'),(8,'DUNGAREES'),(9,'T-SHIRT'),(10,'INFANTS'),
(11,'TROUSERS'),(12,'JOGGERS'),(13,'FLEECE CARDIGAN'),(14,'BERMUDA SHORTS'),(15,'BOMBER JACKET'),(16,'VEST CARDIGAN'),(17,'SWEATSHIRT'),(18,'BLOUSE SHORT SLEEVE'),(19,'SLEEVELESS JACKET'),(20,'SET'),
(21,'JOGGERS FROM TRACKSUIT'),(22,'TRACKSUITS'),(23,'BLOUSE LONG SLEEVE'),(24,'CARDIGAN'),(25,'DRESS'),(26,'LEGGINGS'),(27,'LEGGINGS SET'),(28,'TOP'),(29,'BOLERO'),(30,'SKIRT SET'),
(31,'SKIRT'),(32,'TROUSERS SET'),(33,'CULOTTES'),(34,'PLAYSUIT'),(35,'BLOUSE SWEATER'),(36,'TRENCHCOAT'),(37,'BLOUSE'),(38,'SLEEVELESS BLOUSE'),(39,'JUMPSUIT'),(40,'TUNIC'),
(41,'SWIMMING SUIT BERMUDA'),(42,'PYJAMAS'),(43,'SWIMMING SUIT'),(44,'SWIMMY SET'),(45,'BEACH SHORTS'),(46,'SWIMMY BLOUSE'),(47,'SWIMSUIT'),(48,'BIKINI'),(49,'MONOKINI'),(50,'BLAZER'),
(51,'VEST'),(52,'CARDIGAN SWEATER');

-- 2) Table NECKDESIGN
CREATE TABLE NRGSearch.dbo.NECKDESIGN (
    NeckDesignID INT,
	NeckDesign VARCHAR(100),
	CONSTRAINT PK_NNeckDesign PRIMARY KEY (NeckDesignID)
);
INSERT INTO NRGSearch.dbo.NECKDESIGN (NeckDesignID,NeckDesign)
VALUES 
(1,'ROUND NECK'),(2,'COLLAR'),(3,'TURTLENECK'),(4,'HOODED'),(5,'V NECK'),(6,'OFF SHOULDER'),(7,'HALTERNECK');

-- 3) Table PRODUCTCATEGORY
CREATE TABLE NRGSearch.dbo.PRODUCTCATEGORY (
    ProductCategoryID INT,
	ProductCategory VARCHAR(100),
	CONSTRAINT PK_NProductCategory PRIMARY KEY (ProductCategoryID)

);
INSERT INTO NRGSearch.dbo.PRODUCTCATEGORY (ProductCategoryID,ProductCategory)
VALUES 
(1,'SET'),(2,'BERMUDAS-SHORTS'),(3,'BLOUSES'),(4,'TROUSERS'),(5,'ROMPER'),(6,'SHIRTS'),(7,'OUTDOOR CLOTHING OR COAT'),(8,'CARDIGAN'),(9,'TRACKSUIT'),(10,'DRESS'),
(11,'LEGGINGS'),(12,'SKIRT'),(13,'SWIMMING SUITS'),(14,'PYJAMAS');

-- 4) Table LIFESTAGE
CREATE TABLE NRGSearch.dbo.LIFESTAGE (
    LifeStageID INT,
	LifeStage VARCHAR(100),
	CONSTRAINT PK_NLifeStage PRIMARY KEY (LifeStageID)

);
INSERT INTO NRGSearch.dbo.LIFESTAGE (LifeStageID,LifeStage)
VALUES 
(1,'Infant'),(2,'0 to 5'),(3,'1 to 5'),(4,'1 to 16'),(5,'6 to 16');

-- 5) Table TRENDTHEME
CREATE TABLE NRGSearch.dbo.TRENDTHEME (
    TrendThemeID INT,
	TrendTheme VARCHAR(100),
	CONSTRAINT PK_NTrendTheme PRIMARY KEY (TrendThemeID)

);
INSERT INTO NRGSearch.dbo.TRENDTHEME (TrendThemeID,TrendTheme)
VALUES 
(1,'BASIC LINE'),(2,'BEACH & SPORT'),(3,'BEACH AND SPORT'),(4,'COOL GUYS'),(5,'FREE LIFE'),(6,'ALL SUMMER LONG'),(7,'SAIL AWAY'),(8,'IN ACTION'),(9,'SAILOR SPIRIT'),(10,'WILD GIRL'),
(11,'CHERRIES'),(12,'ICE CREAM'),(13,'FLOWERS'),(14,'BOHO SPIRIT'),(15,'FRENCH RIVIERA'),(16,'BOHO DREAM'),(17,'CALIFORNIA DREAMING'),(18,'WATERMELON'),(19,'PRAIRIE DAYS'),(20,'#SELFIE'),
(21,'DAY DREAMER'),(22,'BE LOVED'),(23,'FASHION-CHIC'),(24,'PROMOTION'),(25,'SPORT LINE'),(26,'SWIMMY'),(27,'SWEET DREAMS'),(28,'BOUTIQUE');

-- 6) Table FIT
CREATE TABLE NRGSearch.dbo.FIT (
    FitID INT,
	Fit VARCHAR(100),
	CONSTRAINT PK_NFit PRIMARY KEY (FitID)
);
INSERT INTO NRGSearch.dbo.FIT (FitID,Fit)
VALUES 
(1,'REGULAR FIT'),(2,'CARGO'),(3,'RELAXED FIT'),(4,'SLIM FIT'),(5,'CHINOS'),(6,'BIKER');

-- 7) Table INSPIRATIONBACKGROUND
CREATE TABLE NRGSearch.dbo.INSPIRATIONBACKGROUND (
    InspirationBackgroundID INT,
	InspirationBackground VARCHAR(100),
	CONSTRAINT PK_NInspirationBackground PRIMARY KEY (InspirationBackgroundID)
);
INSERT INTO NRGSearch.dbo.INSPIRATIONBACKGROUND (InspirationBackgroundID,InspirationBackground)
VALUES 
(1,'CHILDRENSALON'),(2,'PINTEREST'),(3,'STYLE RIGHT'),(4,'SHUTTERSTOCK'),(5,'PITTI IMAGINE'),(6,'NEXT LOOK'),(7,'RETAIL SHOPS');

-- 8) Table SAMPLEMANUFACTURER
CREATE TABLE NRGSearch.dbo.SAMPLEMANUFACTURER (
    SampleManufacturerID INT,
	SampleManufacturer VARCHAR(100),
	CONSTRAINT PK_NSampleManufacturer PRIMARY KEY (SampleManufacturerID)
);
INSERT INTO NRGSearch.dbo.SAMPLEMANUFACTURER (SampleManufacturerID,SampleManufacturer)
VALUES 
(1,'APPLE APPARELSS'),(2,'NULL'),(3,'JESSE GARMENTS LTD'),(4,'BEBESAN TEKS.SAV.VE DIS TIC.LTD.STI.'),(5,'XIAMEN MICROUNION IND.'),(6,'V2 KNITS'),(7,'GDM WEAR LIMITED'),(8,'LANDMARK KNITWEAR'),(9,'JIN SI XUAN IMP AND EXP COMPANY LTD'),(10,'M/S VECTOR INDIA'),
(11,'OGGO EXPORTS'),(12,'YZ TEKSTIL KON.SAN.TEC.LTD.STI'),(13,'WEIHAI CELINE INTER TIO L CO'),(14,'MINIDUNYA TEKSTIL HAKNUR BEBE'),(15,'LUCNE GARMENTS CO.,LTD'),(16,'ISMAIL BASA TEKSTIL  KIS SAN VE TIC LTD'),(17,'CASUAL CLOTHING'),(18,'GLADIOLUS FASHION WEAR LTD'),(19,'AINDHA  KNITWEARS'),(20,'LAKSHARA GARMENTS'),
(21,'LACHMI S INT LTD'),(22,'SIMEX SOURCING LIMITED'),(23,'INDIGO BYING SERVICES LTD'),(24,'AKBEYAZ TEKSTIL KONF.SAN.TIC.LTD.STI.'),(25,'PRA V FASHIONS'),(26,'LAKIDS GIYIM SAN.VE TIC.'),(27,'SAGA INDUSTRIAL LIMITED'),(28,'SHISHI XIAOGALA DRESS AND WEAVING CO LTD'),(29,'���������� ������ & ��� ��'),(30,'FANGLE EXPORT'),
(31,'WPS SRL'),(32,'OZGU MODA TEKSTIL SAN VE TIC LTD STI'),(33,'HANGHOU LI N HUARUI CLOTHING CO LTD'),(34,'AMMIR BABY UNO SRL'),(35,'MELCAN GROUP'),(36,'JIAXING LA LORI FASHION CO.,LTD'),(37,'XIAMEN KINGLAND CO LTD (C& D)'),(38,'MAVERA (LOVETTI)TEKSTIL KUY. GIDA(LOVETTI)'),(39,'CICEK TEKSTIL VE KONF.URUNLERI LTD'),(40,'KVS TEKSTIL LTD STI'),
(41,'FRANCA BABY SRL'),(42,'KND ASSOCIATES'),(43,'COCONUDI  PRINCIPITO LIZZI SRL'),(44,'EMF TEKSTIL KONFEKSIYON SAN VE TIC ANONIM SIRKETI'),(45,'A.H.ATALAY TEKSTIL KONFEKSIYON SA Y LTD STL'),(46,'GAMZELIM  KIDSWEAR');

-- 9) Table LENGTH
CREATE TABLE NRGSearch.dbo.LENGTH (
    LengthID INT,
	Length VARCHAR(100),
	CONSTRAINT PK_NLength PRIMARY KEY (LengthID)
);
INSERT INTO NRGSearch.dbo.LENGTH (LengthID,Length)
VALUES 
(1,'SHORT LENGTH'),(2,'LONG'),(3,'MEDIUM LENGTH'),(4,'KNEE LENGTH'),(5,'CAPRI'),(6,'3/4 LENGTH');

-- 10) Table PRODUCTIONMANUFACTURER
CREATE TABLE NRGSearch.dbo.PRODUCTIONMANUFACTURER (
    ProductionManufacturerID INT,
	ProductionManufacturer VARCHAR(100),
	CONSTRAINT PK_NProductionManufacturer PRIMARY KEY (ProductionManufacturerID)
);
INSERT INTO NRGSearch.dbo.PRODUCTIONMANUFACTURER (ProductionManufacturerID,ProductionManufacturer)
VALUES 
(1,'APPLE APPARELSS'),(2,'NULL'),(3,'JESSE GARMENTS LTD'),(4,'BEBESAN TEKS.SAV.VE DIS TIC.LTD.STI.'),(5,'MINIDUNYA TEKSTIL HAKNUR BEBE'),(6,'CASUAL CLOTHING'),(7,'LANDMARK KNITWEAR'),(8,'V2 KNITS'),(9,'GDM WEAR LIMITED'),(10,'JIN SI XUAN IMP AND EXP COMPANY LTD'),
(11,'PRANAV FASHIONS'),(12,'AINDHA KNITWEARS'),(13,'OGGO EXPORTS'),(14,'M/S VECTOR INDIA'),(15,'YZ TEKSTIL KON.SAN.TEC.LTD.STI'),(16,'WEIHAI CELINE INTERNATIONAL CO'),(17,'ISMAIL BASA TEKSTIL  KIS SAN VE TIC LTD'),(18,'GLADIOLUS FASHION WEAR LTD'),(19,'LACHMI S INT LTD'),(20,'LAKIDS GIYIM SAN.VE TIC.'),
(21,'SAGA INDUSTRIAL LIMITED'),(22,'ΒΑΤΑΝΣΕΒΕΡ ΑΙΝΟΥΡ & ΣΙΑ ΟΕ'),(23,'OZGU MODA TEKSTIL SAN VE TIC LTD STI'),(24,'MELCAN GROUP'),(25,'JIAXING LA LORI FASHION CO.,LTD'),(26,'MAVERA (LOVETTI)TEKSTIL KUY. GIDA(LOVETTI)'),(27,'CICEK TEKSTIL VE KONF.URUNLERI LTD'),(28,'KVS TEKSTIL LTD STI'),(29,'XIAMEN MICROUNION IND.'),(30,'FANGLE EXPORT'),
(31,'EMF TEKSTIL KONFEKSIYON SAN VE TIC ANONIM SIRKETI'),(32,'A.H.ATALAY TEKSTIL KONFEKSIYON SA Y LTD STL'),(33,'GAMZELIM  KIDSWEAR');

-- 11) Table SLEEVE
CREATE TABLE NRGSearch.dbo.SLEEVE (
    SleeveID INT,
	Sleeve VARCHAR(100),
	CONSTRAINT PK_NSleeve PRIMARY KEY (SleeveID)
);
INSERT INTO NRGSearch.dbo.SLEEVE (SleeveID,Sleeve)
VALUES 
(1,'SHORT SLEEVE'),(2,'LONG SLEEVE'),(3,'TURN UP SLEEVE'),(4,'SLEEVELESS'),(5,'RAGLAN SLEEVE'),(6,'CUP SLEEVE'),(7,'3/4 FLARED'),(8,'FLARED');

-- 12) Table COLLARDESIGN
CREATE TABLE NRGSearch.dbo.COLLARDESIGN (
    CollarDesignID INT,
	CollarDesign VARCHAR(100),
	CONSTRAINT PK_NCollarDesign PRIMARY KEY (CollarDesignID)
);
INSERT INTO NRGSearch.dbo.COLLARDESIGN (CollarDesignID,CollarDesign)
VALUES 
(1,'POLO COLLAR'),(2,'SHIRT COLLAR'),(3,'FLAT KNITTED RIB'),(4,'MAO COLLAR'),(5,'STAND UP COLLAR');

-- 13) Table GENDER
CREATE TABLE NRGSearch.dbo.GENDER (
    GenderID INT,
	Gender VARCHAR(100),
	CONSTRAINT PK_NGender PRIMARY KEY (GenderID)
);
INSERT INTO NRGSearch.dbo.GENDER (GenderID,Gender)
VALUES 
(1,'Man-woman'),(2,'Boy-Girl');

-- ################################# Table PRODUCT ############################################### --
CREATE TABLE NRGSearch.dbo.PRODUCT (
	ProductNo INT IDENTITY(1,1) NOT NULL,
	ProductCode VARCHAR(40) NOT NULL,
	ProductTitle VARCHAR(512), 
	ForeignDescription NVARCHAR(512), 
	Composition VARCHAR(2000),
	ForeignComposition NVARCHAR(2000),
    ProductCategoryID INT, 
	ProductSubcategoryID INT, 
	Season VARCHAR(30), 
	BusinessUnit VARCHAR(30), 
	GenderID INT,
	LifeStageID INT,
	TrendThemeID INT,
	InspirationBackgroundID INT,
	Sizeset VARCHAR(64), 
	LengthID INT,
	SleeveID INT, 
	CollarDesignID INT, 
	NeckDesignID INT,
	FitID INT, 
	[Fabric/AccessoriesDetails] VARCHAR(1) NOT NULL,
	SampleManufacturerID INT,
	SamplePrice DECIMAL(10,2), 
	ProductionManufacturerID INT, 
	ProductionPrice DECIMAL(38,9),
	WholesalePrice DECIMAL(10,6),
	RetailPrice DECIMAL(10,6), 
	Photo NVARCHAR(254),
	Sketch NVARCHAR(254), 
	Colors NVARCHAR(MAX), 
	ClusterID INT NOT NULL, 
	FClusterID INT NOT NULL,
	CONSTRAINT PK_NPruct PRIMARY KEY (ProductNo)
);

-- ################################# Reconstruct Table PRODUCTSOLD in 3rd Normal Form (Intermediate Tables) --- Table PRODUCTS ############################################### --
INSERT INTO NRGSearch.dbo.PRODUCT
SELECT PROD.ProductNo, PROD.ProductTitle, PROD.ForeignDescription, PROD.Composition, PROD.ForeignComposition,
       PRODCAT.ProductCategoryID, PRODSUB.ProductSubcategoryID, PROD.Season, PROD.BusinessUnit, GENDER.GenderID,
	   LIFE.LifeStageID, TREND.TrendThemeID, INSPIRATION.InspirationBackgroundID, PROD.Sizeset, LENGTH.LengthID,
	   SLEEVE.SleeveID, COLLAR.CollarDesignID, NECK.NeckDesignID, FIT.FitID, PROD.[Fabric/AccessoriesDetails],
	   SAMPLE.SampleManufacturerID, PROD.SamplePrice, PRODUCTION.ProductionManufacturerID, PROD.ProductionPrice,
	   PROD.WholesalePrice, PROD.RetailPrice, PROD.Photo, PROD.Sketch, PROD.Colors, PROD.ClusterID, PROD.FClusterID
FROM (NRGSearch.dbo.PRODUCTSOLD AS PROD  -- (1)
LEFT JOIN NRGSearch.dbo.PRODUCTSUBCATEGORY AS PRODSUB
ON PROD.ProductSubcategory = PRODSUB.ProductSubcategory
LEFT JOIN NRGSearch.dbo.NECKDESIGN AS NECK -- (2)
ON PROD.NeckDesign = NECK.NeckDesign
LEFT JOIN NRGSearch.dbo.PRODUCTCATEGORY AS PRODCAT -- (3)
ON PROD.ProductCategory = PRODCAT.ProductCategory
LEFT JOIN NRGSearch.dbo.LIFESTAGE AS LIFE -- (4)
ON PROD.LifeStage = LIFE.LifeStage
LEFT JOIN NRGSearch.dbo.TRENDTHEME AS TREND -- (5)
ON PROD.TrendTheme = TREND.TrendTheme
LEFT JOIN NRGSearch.dbo.FIT AS FIT -- (6)
ON PROD.Fit = FIT.Fit
LEFT JOIN NRGSearch.dbo.INSPIRATIONBACKGROUND AS INSPIRATION -- (7)
ON PROD.InspirationBackground = INSPIRATION.InspirationBackground
LEFT JOIN NRGSearch.dbo.SAMPLEMANUFACTURER AS SAMPLE -- (8)
ON PROD.SampleManufacturer = SAMPLE.SampleManufacturer
LEFT JOIN NRGSearch.dbo.LENGTH AS LENGTH -- (9)
ON PROD.Length = LENGTH.Length
LEFT JOIN NRGSearch.dbo.PRODUCTIONMANUFACTURER AS PRODUCTION -- (10)
ON PROD.ProductionManufacturer = PRODUCTION.ProductionManufacturer
LEFT JOIN NRGSearch.dbo.SLEEVE AS SLEEVE -- (11)
ON PROD.Sleeve = SLEEVE.Sleeve
LEFT JOIN NRGSearch.dbo.COLLARDESIGN AS COLLAR -- (12)
ON PROD.CollarDesign = COLLAR.CollarDesign
LEFT JOIN NRGSearch.dbo.GENDER AS GENDER -- (13)
ON PROD.Gender = GENDER.Gender)

-- ########################################## Make NULL Values of ID Columns -1 ###################################################
UPDATE NRGSearch.dbo.PRODUCT
SET SleeveID = -1
WHERE SleeveID IS NULL

UPDATE NRGSearch.dbo.PRODUCT
SET TrendThemeID = -1
WHERE TrendThemeID IS NULL

UPDATE NRGSearch.dbo.PRODUCT
SET LengthID = -1
WHERE LengthID IS NULL

UPDATE NRGSearch.dbo.PRODUCT
SET CollarDesignID = -1
WHERE CollarDesignID IS NULL

UPDATE NRGSearch.dbo.PRODUCT
SET NeckDesignID = -1
WHERE NeckDesignID IS NULL

UPDATE NRGSearch.dbo.PRODUCT
SET FitID = -1
WHERE FitID IS NULL

UPDATE NRGSearch.dbo.PRODUCT
SET SampleManufacturerID = -1
WHERE SampleManufacturerID IS NULL

UPDATE NRGSearch.dbo.PRODUCT
SET ProductionManufacturerID = -1
WHERE ProductionManufacturerID IS NULL

UPDATE NRGSearch.dbo.PRODUCT
SET NeckDesignID = -1
WHERE NeckDesignID IS NULL

ALTER TABLE NRGSearch.dbo.PRODUCT
ALTER COLUMN Sketch nvarchar(254);

-- ################################# Table COLOR ############################################### --
CREATE TABLE NRGSearch.dbo.COLOR (
	ProductNo INT NOT NULL,
    ColorID INT NOT NULL,
    Percentage FLOAT,
	Ranking INT NOT NULL,
	CONSTRAINT PK_NColor PRIMARY KEY (ProductNo, Ranking)
);

-- ################################# Table COLORRGB ############################################### --
CREATE TABLE NRGSearch.dbo.COLORRGB (
	ColorID INT NOT NULL,
    Red INT,
    Green INT,
    Blue INT,
	Label VARCHAR(40),
	LabelDetailed VARCHAR(40),
	CONSTRAINT PK_NColorRGB PRIMARY KEY (ColorID)
);

-- ################################# Table CLUSTER ############################################### --
CREATE TABLE NRGSearch.dbo.CLUSTER (
	ClusterID INT NOT NULL DEFAULT -1,
    CentSamplePrice INT,
    CentProductCategoryID INT,
	CentProductSubcategoryID INT,
	CentGenderID INT,
	CentLifeStageID INT,
	CentLengthID INT,
	CentSleeveID INT,
	CentCollarDesignID INT,
    CentNeckDesignID INT,
	CentFitID INT,
	CONSTRAINT PK_NCluster PRIMARY KEY (ClusterID)
);

-- ################################# Table FINANCIALCLUSTER ############################################### --
CREATE TABLE NRGSearch.dbo.FINANCIALCLUSTER (
	FClusterID INT NOT NULL DEFAULT -1,
	CONSTRAINT PK_NFinancialCluster PRIMARY KEY (FClusterID)
);

-- ################################# Table SEARCH ############################################### --
CREATE TABLE NRGSearch.dbo.SEARCH (
	SearchID INT IDENTITY(1,1) NOT NULL,
	SearchSession INT NOT NULL,
	UserID INT,
	RoundID INT,
	CombinationID INT,
	SocialMediaID INT,
    Timestamp DATETIME NULL DEFAULT GETDATE(),
    Season VARCHAR(40),
    TimeThreshold DATETIME,
	NumOfImages INT DEFAULT(30),
	TrendsPer INT DEFAULT(25),
	RelevancyPer INT DEFAULT(25),
	CompDBPer INT DEFAULT(25),
	SalesPer INT DEFAULT(25),
	CONSTRAINT PK_NSearch PRIMARY KEY (SearchID)
 );


-- ################################# Table RESULT ############################################### --
CREATE TABLE NRGSearch.dbo.RESULT (
	SearchID INT NOT NULL,
	UserID INT NOT NULL,
    ProductNo INT,
	DBID INT,
    Click BIT DEFAULT(0),
    GradeSystem INT,
	GradeUser INT,
	isFavorite BIT DEFAULT(0),
	CONSTRAINT PK_NResult PRIMARY KEY (SearchID, UserID, ProductNo, DBID)
 );

-- ################################# Table DASHBOARD ############################################### --
CREATE TABLE NRGSearch.dbo.DASHBOARD (
    UserID INT,
	ProductNo INT,
	DBID INT,
	Timestamp DATETIME NULL DEFAULT GETDATE(),
    DashboardName VARCHAR(100),
	CONSTRAINT PK_NDashboard PRIMARY KEY (UserID, ProductNo, DBID, DashboardName)
 );

 -- Run the script Combination_Table.py from Database folder in order to construct the COMBINATION table

 -- ################################# Table COMBINATION ############################################### --
ALTER TABLE NRGSearch.dbo.COMBINATION
ALTER COLUMN CombinationID INT NOT NULL

ALTER TABLE NRGSearch.dbo.COMBINATION
ADD	CONSTRAINT PK_NSCombination PRIMARY KEY (CombinationID);

-- ################################# Table COMMENT ############################################### --
CREATE TABLE NRGSearch.dbo.COMMENT (
    UserID INT,
	ProductNo INT,
	DBID INT,
	Comment VARCHAR(100),
	CONSTRAINT PK_NComment PRIMARY KEY (UserID, ProductNo, DBID)
 );

 -- ################################# Table RESTRICTION ############################################### --
CREATE TABLE NRGSearch.dbo.RESTRICTION (
    UserID INT,
	CombinationID INT,
	MatchID INT,
	isIncluded BIT DEFAULT(0),
	CONSTRAINT PK_NRestriction PRIMARY KEY (UserID, CombinationID, MatchID)
 );


 -- ################################# Table USERS ############################################### --
 CREATE TABLE NRGSearch.dbo.USERS (
    UserID INT,
	Username VARCHAR(40),
    Password VARCHAR(40),
    Email VARCHAR(40),
	Company VARCHAR(40),
    Customer VARCHAR(40),
	CONSTRAINT PK_NUsers PRIMARY KEY (UserID)
 );

 -- ################################# Table FINANCE ############################################### --
-- SELECT *
-- INTO NRGSearch.dbo.FINANCE
-- FROM NRG.dbo.FINANCE

-- ################################# FOREIGN KEYS ############################################### --
-- PRODUCT
-- #A
INSERT INTO NRGSearch.dbo.CLUSTER (ClusterID,CentSamplePrice,CentProductCategoryID,CentProductSubcategoryID,CentGenderID,CentLifeStageID,CentLengthID,CentSleeveID,CentCollarDesignID,CentNeckDesignID,CentFitID)
VALUES (-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
	   (0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
	   (1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
	   (2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
	   (3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
	   (4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
	   (5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

ALTER TABLE NRGSearch.dbo.PRODUCT
ADD CONSTRAINT FK_NProduct_NCluster FOREIGN KEY (ClusterID) REFERENCES NRGSearch.dbo.CLUSTER(ClusterID);
-- #B
INSERT INTO NRGSearch.dbo.FINANCIALCLUSTER(FClusterID)
VALUES (-1);
ALTER TABLE NRGSearch.dbo.PRODUCT
ADD CONSTRAINT FK_NProduct_NFinancialCluster FOREIGN KEY (FClusterID) REFERENCES NRGSearch.dbo.FINANCIALCLUSTER(FClusterID);
-- #1
INSERT INTO NRGSearch.dbo.PRODUCTSUBCATEGORY(ProductSubcategoryID,ProductSubcategory)
VALUES (-1,NULL);
ALTER TABLE NRGSearch.dbo.PRODUCT
ADD CONSTRAINT FK_NProduct_NProductSubcategory FOREIGN KEY (ProductSubcategoryID) REFERENCES NRGSearch.dbo.PRODUCTSUBCATEGORY(ProductSubcategoryID);
-- #2
INSERT INTO NRGSearch.dbo.NECKDESIGN(NeckDesignID,NeckDesign)
VALUES (-1,NULL);
ALTER TABLE NRGSearch.dbo.PRODUCT
ADD CONSTRAINT FK_NProduct_NNeckDesign FOREIGN KEY (NeckDesignID) REFERENCES NRGSearch.dbo.NECKDESIGN(NeckDesignID);
-- #3
INSERT INTO NRGSearch.dbo.PRODUCTCATEGORY(ProductCategoryID,ProductCategory)
VALUES (-1,NULL);
ALTER TABLE NRGSearch.dbo.PRODUCT
ADD CONSTRAINT FK_NProduct_NProductCategory FOREIGN KEY (ProductCategoryID) REFERENCES NRGSearch.dbo.PRODUCTCATEGORY(ProductCategoryID);
-- #4
INSERT INTO NRGSearch.dbo.LIFESTAGE(LifeStageID,LifeStage)
VALUES (-1,NULL);
ALTER TABLE NRGSearch.dbo.PRODUCT
ADD CONSTRAINT FK_NProduct_NLifeStage FOREIGN KEY (LifeStageID) REFERENCES NRGSearch.dbo.LIFESTAGE(LifeStageID);
-- #5
INSERT INTO NRGSearch.dbo.TRENDTHEME(TrendThemeID,TrendTheme)
VALUES (-1,NULL);
ALTER TABLE NRGSearch.dbo.PRODUCT
ADD CONSTRAINT FK_NProduct_NTrendTheme FOREIGN KEY (TrendThemeID) REFERENCES NRGSearch.dbo.TRENDTHEME(TrendThemeID);
-- #6
INSERT INTO NRGSearch.dbo.FIT(FitID,Fit)
VALUES (-1,NULL);
ALTER TABLE NRGSearch.dbo.PRODUCT
ADD CONSTRAINT FK_NProduct_NFit FOREIGN KEY (FitID) REFERENCES NRGSearch.dbo.FIT(FitID);
-- #7
INSERT INTO NRGSearch.dbo.INSPIRATIONBACKGROUND(InspirationBackgroundID,InspirationBackground)
VALUES (-1,NULL);
ALTER TABLE NRGSearch.dbo.PRODUCT
ADD CONSTRAINT FK_NProduct_NInspirationBackground FOREIGN KEY (InspirationBackgroundID) REFERENCES NRGSearch.dbo.INSPIRATIONBACKGROUND(InspirationBackgroundID);
-- #8
INSERT INTO NRGSearch.dbo.SAMPLEMANUFACTURER(SampleManufacturerID,SampleManufacturer)
VALUES (-1,NULL);
ALTER TABLE NRGSearch.dbo.PRODUCT
ADD CONSTRAINT FK_NProduct_NSampleManufacturer FOREIGN KEY (SampleManufacturerID) REFERENCES NRGSearch.dbo.SAMPLEMANUFACTURER(SampleManufacturerID);
-- #9
INSERT INTO NRGSearch.dbo.LENGTH(LengthID,Length)
VALUES (-1,NULL);
ALTER TABLE NRGSearch.dbo.PRODUCT
ADD CONSTRAINT FK_NProduct_NLength FOREIGN KEY (LengthID) REFERENCES NRGSearch.dbo.LENGTH(LengthID);
-- #10
INSERT INTO NRGSearch.dbo.PRODUCTIONMANUFACTURER(ProductionManufacturerID,ProductionManufacturer)
VALUES (-1,NULL);
ALTER TABLE NRGSearch.dbo.PRODUCT
ADD CONSTRAINT FK_NProduct_NProductionManufacturer FOREIGN KEY (ProductionManufacturerID) REFERENCES NRGSearch.dbo.PRODUCTIONMANUFACTURER(ProductionManufacturerID);
-- #11
INSERT INTO NRGSearch.dbo.SLEEVE(SleeveID,Sleeve)
VALUES (-1,NULL);
ALTER TABLE NRGSearch.dbo.PRODUCT
ADD CONSTRAINT FK_NProduct_NSleeve FOREIGN KEY (SleeveID) REFERENCES NRGSearch.dbo.SLEEVE(SleeveID);
-- #12
INSERT INTO NRGSearch.dbo.COLLARDESIGN(CollarDesignID,CollarDesign)
VALUES (-1,NULL);
ALTER TABLE NRGSearch.dbo.PRODUCT
ADD CONSTRAINT FK_NProduct_NCollarDesign FOREIGN KEY (CollarDesignID) REFERENCES NRGSearch.dbo.COLLARDESIGN(CollarDesignID);
-- #13
INSERT INTO NRGSearch.dbo.GENDER(GenderID,Gender)
VALUES (-1,NULL);
ALTER TABLE NRGSearch.dbo.PRODUCT
ADD CONSTRAINT FK_NProduct_NGender FOREIGN KEY (GenderID) REFERENCES NRGSearch.dbo.GENDER(GenderID);

-- COLOR
ALTER TABLE NRGSearch.dbo.COLOR
ADD CONSTRAINT FK_NColor_NProduct FOREIGN KEY (ProductNo) REFERENCES NRGSearch.dbo.PRODUCT(ProductNo);
ALTER TABLE NRGSearch.dbo.COLOR
ADD CONSTRAINT FK_NColor_NColorRGB FOREIGN KEY (ColorID) REFERENCES NRGSearch.dbo.COLORRGB(ColorID);

-- SEARCH
ALTER TABLE NRGSearch.dbo.SEARCH
ADD CONSTRAINT FK_NSSearch_NSUsers  FOREIGN KEY (UserID) REFERENCES NRGSearch.dbo.USERS(UserID)

ALTER TABLE NRGSearch.dbo.SEARCH
ADD CONSTRAINT FK_NSSearch_NSCombination  FOREIGN KEY (CombinationID) REFERENCES NRGSearch.dbo.COMBINATION(CombinationID)

-- RESULT
ALTER TABLE NRGSearch.dbo.RESULT
ADD CONSTRAINT FK_NSResult_NSUsers FOREIGN KEY (UserID) REFERENCES NRGSearch.dbo.USERS(UserID)

ALTER TABLE NRGSearch.dbo.RESULT
ADD CONSTRAINT FK_NSResult_NSSearch FOREIGN KEY (SearchID) REFERENCES NRGSearch.dbo.SEARCH(SearchID)

-- DASHBOARD
ALTER TABLE NRGSearch.dbo.DASHBOARD
ADD CONSTRAINT FK_NSDashboard_NSUsers  FOREIGN KEY (UserID) REFERENCES NRGSearch.dbo.USERS(UserID)

-- COMMENT
ALTER TABLE NRGSearch.dbo.COMMENT
ADD CONSTRAINT FK_NSComment_NSUsers  FOREIGN KEY (UserID) REFERENCES NRGSearch.dbo.USERS(UserID)

-- RESTRICTION
ALTER TABLE NRGSearch.dbo.RESTRICTION
ADD CONSTRAINT FK_NSRestriction_NSUsers  FOREIGN KEY (UserID) REFERENCES NRGSearch.dbo.USERS(UserID)

ALTER TABLE NRGSearch.dbo.RESTRICTION
ADD CONSTRAINT FK_NSRestriction_NSCombination  FOREIGN KEY (CombinationID) REFERENCES NRGSearch.dbo.COMBINATION(CombinationID)

/* #################################       2        ################################################
   ################################# SocialMedia DB ################################################ */

-- Create Database
CREATE DATABASE SocialMedia;
-- Drop Database (BE CAREFUL)
-- DROP DATABASE SocialMedia

-- ################################# Table PRODUCT ############################################### --
CREATE TABLE SocialMedia.dbo.PRODUCT (
	ProductNo INT IDENTITY(1,1) NOT NULL,
	Crawler VARCHAR(40) NOT NULL,
	UserIn VARCHAR(128) NOT NULL DEFAULT HOST_NAME(),
    SearchWords VARCHAR(500),
    Image NVARCHAR(500),
	ImageBlob VARBINARY(MAX),
	url VARCHAR(500),
	ImageSource VARCHAR(500),
	SiteClothesHeadline VARCHAR(500),
    Color VARCHAR(40),
	GenderID INT,
	Brand VARCHAR(40),
    Metadata VARCHAR(2000),
	ProductCategoryID INT,
	ProductSubcategoryID INT,
	LengthID INT,
	SleeveID INT,
	CollarDesignID INT,
    NeckDesignID INT,
	FitID INT,
	ClusterID INT NOT NULL DEFAULT -1,
	FClusterID INT NOT NULL DEFAULT -1,
	CONSTRAINT PK_SNProduct PRIMARY KEY (ProductNo)
);

-- ################################# Table PRODUCTHISTORY ############################################### --
CREATE TABLE SocialMedia.dbo.PRODUCTHISTORY (
	ProductNo INT NOT NULL,
	SearchDate DATETIME NOT NULL DEFAULT GETDATE(),
	UserIn VARCHAR(128) NOT NULL DEFAULT HOST_NAME(),
    ReferenceOrder INT DEFAULT -1,
	TrendingOrder INT DEFAULT -1,
	Price FLOAT,
	CONSTRAINT PK_SNProducthistory PRIMARY KEY (ProductNo, SearchDate)
);
 -- ################################# Table FASHIONBOOKS ############################################### --
CREATE TABLE SocialMedia.dbo.FASHIONBOOKS (
    ProductNo INT IDENTITY(1,1) NOT NULL,
	Crawler VARCHAR(40) NOT NULL,
	SearchDate DATETIME NULL DEFAULT GETDATE(),
	UserIn VARCHAR(128) NOT NULL DEFAULT HOST_NAME(),
    SearchWords VARCHAR(500),
    Image NVARCHAR(500),
	ImageBlob VARBINARY(MAX),
	ImageSource VARCHAR(500),
    Color VARCHAR(40),
	GenderID INT,
	Brand VARCHAR(40),
    Metadata VARCHAR(2000),
	ProductCategoryID INT,
	ProductSubcategoryID INT,
	LengthID INT,
	SleeveID INT,
	CollarDesignID INT,
    NeckDesignID INT,
	FitID INT,
	ClusterID INT NOT NULL DEFAULT -1,
	FClusterID INT NOT NULL DEFAULT -1,
	CONSTRAINT PK_SNFashionbooks PRIMARY KEY (ProductNo)
 );

--INSERT INTO SocialMedia.dbo.4 (ProductNo,SearchDate,SearchWords,Image,ListingOrder,url,ImageSource,SiteClothesHeadline,Color,Gender,Brand,CurrentPrice,InitialPrice,Metadata)
--SELECT P.ProductNo, P.SearchDate,P.SearchWords,P.Image,P.ListingOrder,P.url,P.ImageSource,P.SiteClothesHeadline,P.Color,P.Gender,P.Brand,P.CurrentPrice,P.InitialPrice,P.Metadata
--FROM SocialMedia.dbo.PRODUCTS AS P

-- ################################# Create INTERMEDIATE Tables ############################################### --
-- 1) Table PRODUCTSUBCATEGORY
CREATE TABLE SocialMedia.dbo.PRODUCTSUBCATEGORY (
    ProductSubcategoryID INT,
	ProductSubcategory VARCHAR(100),
	GenProductSubcategoryID INT,
	CONSTRAINT PK_SProductSubcategory PRIMARY KEY (ProductSubcategoryID)
);
INSERT INTO SocialMedia.dbo.PRODUCTSUBCATEGORY (ProductSubcategoryID, ProductSubcategory, GenProductSubcategoryID)
VALUES 
(1,'SHORT SET',-1),(2,'SHORTS',-1),(3,'BLOUSE POLO SHORT SLEEVE',-1),(4,'JEANS',-1),(5,'INFANT ROMPER',-1),(6,'SHIRT',-1),(7,'JACKET',-1),(8,'DUNGAREES',-1),(9,'T-SHIRT',-1),(10,'INFANTS',-1),
(11,'TROUSERS',-1),(12,'JOGGERS',-1),(13,'FLEECE CARDIGAN',-1),(14,'BERMUDA SHORTS',-1),(15,'BOMBER JACKET',-1),(16,'VEST CARDIGAN',-1),(17,'SWEATSHIRT',-1),(18,'BLOUSE SHORT SLEEVE',-1),(19,'SLEEVELESS JACKET',-1),(20,'SET',-1),
(21,'JOGGERS FROM TRACKSUIT',-1),(22,'TRACKSUITS',-1),(23,'BLOUSE LONG SLEEVE',-1),(24,'CARDIGAN',-1),(25,'DRESS',-1),(26,'LEGGINGS',-1),(27,'LEGGINGS SET',-1),(28,'TOP',-1),(29,'BOLERO',-1),(30,'SKIRT SET',-1),
(31,'SKIRT',-1),(32,'TROUSERS SET',-1),(33,'CULOTTES',-1),(34,'PLAYSUIT',-1),(35,'BLOUSE SWEATER',-1),(36,'TRENCHCOAT',-1),(37,'BLOUSE',-1),(38,'SLEEVELESS BLOUSE',-1),(39,'JUMPSUIT',-1),(40,'TUNIC',-1),
(41,'SWIMMING SUIT BERMUDA',-1),(42,'PYJAMAS',-1),(43,'SWIMMING SUIT',-1),(44,'SWIMMY SET',-1),(45,'BEACH SHORTS',-1),(46,'SWIMMY BLOUSE',-1),(47,'SWIMSUIT',-1),(48,'BIKINI',-1),(49,'MONOKINI',-1),(50,'BLAZER',-1),
(51,'VEST',-1),(52,'CARDIGAN SWEATER',-1);

-- 2) Table NECKDESIGN
CREATE TABLE SocialMedia.dbo.NECKDESIGN (
    NeckDesignID INT,
	NeckDesign VARCHAR(100),
	GenNeckDesignID INT,
	CONSTRAINT PK_SNeckDesign PRIMARY KEY (NeckDesignID)
);
INSERT INTO SocialMedia.dbo.NECKDESIGN
VALUES 
(1,'ROUND NECK',-1),(2,'COLLAR',-1),(3,'TURTLENECK',-1),(4,'HOODED',-1),(5,'V NECK',-1),(6,'OFF SHOULDER',-1),(7,'HALTERNECK',-1);

-- 3) Table PRODUCTCATEGORY
CREATE TABLE SocialMedia.dbo.PRODUCTCATEGORY (
    ProductCategoryID INT,
	ProductCategory VARCHAR(100),
	GenProductCategoryID INT,
	CONSTRAINT PK_SProductCategory PRIMARY KEY (ProductCategoryID)

);
INSERT INTO SocialMedia.dbo.PRODUCTCATEGORY
VALUES 
(1,'SET',-1),(2,'BERMUDAS-SHORTS',-1),(3,'BLOUSES',-1),(4,'TROUSERS',-1),(5,'ROMPER',-1),(6,'SHIRTS',-1),(7,'OUTDOOR CLOTHING OR COAT',-1),(8,'CARDIGAN',-1),(9,'TRACKSUIT',-1),(10,'DRESS',-1),
(11,'LEGGINGS',-1),(12,'SKIRT',-1),(13,'SWIMMING SUITS',-1),(14,'PYJAMAS',-1);


-- 4) Table FIT
CREATE TABLE SocialMedia.dbo.FIT (
    FitID INT,
	Fit VARCHAR(100),
	GenFitID INT,
	CONSTRAINT PK_SFit PRIMARY KEY (FitID)
);
INSERT INTO SocialMedia.dbo.FIT
VALUES 
(1,'REGULAR FIT',-1),(2,'CARGO',-1),(3,'RELAXED FIT',-1),(4,'SLIM FIT',-1),(5,'CHINOS',-1),(6,'BIKER',-1);


-- 5) Table LENGTH
CREATE TABLE SocialMedia.dbo.LENGTH (
    LengthID INT,
	Length VARCHAR(100),
	GenLengthID INT,
	CONSTRAINT PK_SLength PRIMARY KEY (LengthID)
);
INSERT INTO SocialMedia.dbo.LENGTH
VALUES 
(1,'SHORT LENGTH',-1),(2,'LONG',-1),(3,'MEDIUM LENGTH',-1),(4,'KNEE LENGTH',-1),(5,'CAPRI',-1),(6,'3/4 LENGTH',-1);

-- 6) Table SLEEVE
CREATE TABLE SocialMedia.dbo.SLEEVE (
    SleeveID INT,
	Sleeve VARCHAR(100),
	GenSleeveID INT,
	CONSTRAINT PK_SSleeve PRIMARY KEY (SleeveID)
);
INSERT INTO SocialMedia.dbo.SLEEVE
VALUES 
(1,'SHORT SLEEVE',-1),(2,'LONG SLEEVE',-1),(3,'TURN UP SLEEVE',-1),(4,'SLEEVELESS',-1),(5,'RAGLAN SLEEVE',-1),(6,'CUP SLEEVE',-1),(7,'3/4 FLARED',-1),(8,'FLARED',-1);

-- 7) Table COLLARDESIGN
CREATE TABLE SocialMedia.dbo.COLLARDESIGN (
    CollarDesignID INT,
	CollarDesign VARCHAR(100),
	GenCollarDesignID INT,
	CONSTRAINT PK_SCollarDesign PRIMARY KEY (CollarDesignID)
);
INSERT INTO SocialMedia.dbo.COLLARDESIGN
VALUES 
(1,'POLO COLLAR',-1),(2,'SHIRT COLLAR',-1),(3,'FLAT KNITTED RIB',-1),(4,'MAO COLLAR',-1),(5,'STAND UP COLLAR',-1);

-- 8) Table GENDER (MAN, WOMAN) ---- DIFFERENCE WITH ENERGIERS GENDER
CREATE TABLE SocialMedia.dbo.GENDER (
    GenderID INT,
	Gender VARCHAR(100),
	CONSTRAINT PK_SGender PRIMARY KEY (GenderID)
);
INSERT INTO SocialMedia.dbo.GENDER
VALUES 
(1,'MAN'),(2,'WOMAN'), (3,'KIDS');

-- ################################# Table COLOR ############################################### --
CREATE TABLE SocialMedia.dbo.COLOR (
	ProductNo INT NOT NULL,
	ColorID INT NOT NULL,
    Percentage FLOAT,
	Ranking INT NOT NULL,
	CONSTRAINT PK_SColor PRIMARY KEY (ProductNo, Ranking)
);

-- ################################# Table COLORRGB ############################################### --
CREATE TABLE SocialMedia.dbo.COLORRGB (
	ColorID INT NOT NULL,
    Red INT,
    Green INT,
    Blue INT,
	Label VARCHAR(40),
	LabelDetailed VARCHAR(40),
	CONSTRAINT PK_SColorRGB PRIMARY KEY (ColorID)
);

-- ################################# Table CLUSTER ############################################### --
CREATE TABLE SocialMedia.dbo.CLUSTER (
	ClusterID INT NOT NULL DEFAULT -1,
    CentSamplePrice INT,
    CentProductCategoryID INT,
	CentProductSubcategoryID INT,
	CentGenderID INT,
	CentLifeStageID INT,
	CentLengthID INT,
	CentSleeveID INT,
	CentCollarDesignID INT,
    CentNeckDesignID INT,
	CentFitID INT,
	CONSTRAINT PK_SCluster PRIMARY KEY (ClusterID)
);

-- ################################# Table FINANCIALCLUSTER ############################################### --
CREATE TABLE SocialMedia.dbo.FINANCIALCLUSTER (
	FClusterID INT NOT NULL DEFAULT -1,
	CONSTRAINT PK_SFinancialCluster PRIMARY KEY (FClusterID)
);

-- ################################# FOREIGN KEYS ############################################### --
/* -- CHECKING 
SELECT TABLE_NAME,COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, IS_NULLABLE
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'PRODUCTS' */

-- #A
INSERT INTO SocialMedia.dbo.CLUSTER (ClusterID,CentSamplePrice,CentProductCategoryID,CentProductSubcategoryID,CentGenderID,CentLifeStageID,CentLengthID,CentSleeveID,CentCollarDesignID,CentNeckDesignID,CentFitID)
VALUES (-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
	   (0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
	   (1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
	   (2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
	   (3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
	   (4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
	   (5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

ALTER TABLE SocialMedia.dbo.PRODUCT
ADD CONSTRAINT FK_SProduct_SCluster FOREIGN KEY (ClusterID) REFERENCES SocialMedia.dbo.CLUSTER(ClusterID);
-- #B
INSERT INTO SocialMedia.dbo.FINANCIALCLUSTER(FClusterID)
VALUES (-1);
ALTER TABLE SocialMedia.dbo.PRODUCT
ADD CONSTRAINT FK_SProduct_SFinancialCluster FOREIGN KEY (FClusterID) REFERENCES SocialMedia.dbo.FINANCIALCLUSTER(FClusterID);
-- #1
INSERT INTO SocialMedia.dbo.PRODUCTSUBCATEGORY(ProductSubcategoryID,ProductSubcategory)
VALUES (-1,NULL);
ALTER TABLE SocialMedia.dbo.PRODUCT
ADD CONSTRAINT FK_SProduct_SProductSubcategory FOREIGN KEY (ProductSubcategoryID) REFERENCES SocialMedia.dbo.PRODUCTSUBCATEGORY(ProductSubcategoryID);
-- #1 FASHIONBOOKS
ALTER TABLE SocialMedia.dbo.FASHIONBOOKS
ADD CONSTRAINT FK_SFashionbooks_SProductSubcategory FOREIGN KEY (ProductSubcategoryID) REFERENCES SocialMedia.dbo.PRODUCTSUBCATEGORY(ProductSubcategoryID);
-- #2
INSERT INTO SocialMedia.dbo.NECKDESIGN(NeckDesignID,NeckDesign)
VALUES (-1,NULL);
ALTER TABLE SocialMedia.dbo.PRODUCT
ADD CONSTRAINT FK_SProduct_SNeckDesign FOREIGN KEY (NeckDesignID) REFERENCES SocialMedia.dbo.NECKDESIGN(NeckDesignID);
-- #2 FASHIONBOOKS
ALTER TABLE SocialMedia.dbo.FASHIONBOOKS
ADD CONSTRAINT FK_SFashionbooks_SNeckDesign FOREIGN KEY (NeckDesignID) REFERENCES SocialMedia.dbo.NECKDESIGN(NeckDesignID);
-- #3
INSERT INTO SocialMedia.dbo.PRODUCTCATEGORY(ProductCategoryID,ProductCategory)
VALUES (-1,NULL);
ALTER TABLE SocialMedia.dbo.PRODUCT
ADD CONSTRAINT FK_SProduct_SProductCategory FOREIGN KEY (ProductCategoryID) REFERENCES SocialMedia.dbo.PRODUCTCATEGORY(ProductCategoryID);
-- #4
INSERT INTO SocialMedia.dbo.FIT(FitID,Fit)
VALUES (-1,NULL);
ALTER TABLE SocialMedia.dbo.PRODUCT
ADD CONSTRAINT FK_SProduct_SFit FOREIGN KEY (FitID) REFERENCES SocialMedia.dbo.FIT(FitID);
-- #5
INSERT INTO SocialMedia.dbo.LENGTH(LengthID,Length)
VALUES (-1,NULL);
ALTER TABLE SocialMedia.dbo.PRODUCT
ADD CONSTRAINT FK_SProduct_SLength FOREIGN KEY (LengthID) REFERENCES SocialMedia.dbo.LENGTH(LengthID);
-- #6
INSERT INTO SocialMedia.dbo.SLEEVE(SleeveID,Sleeve)
VALUES (-1,NULL);
ALTER TABLE SocialMedia.dbo.PRODUCT
ADD CONSTRAINT FK_SProduct_SSleeve FOREIGN KEY (SleeveID) REFERENCES SocialMedia.dbo.SLEEVE(SleeveID);
-- #7
INSERT INTO SocialMedia.dbo.COLLARDESIGN(CollarDesignID,CollarDesign)
VALUES (-1,NULL);
ALTER TABLE SocialMedia.dbo.PRODUCT
ADD CONSTRAINT FK_SProduct_SCollarDesign FOREIGN KEY (CollarDesignID) REFERENCES SocialMedia.dbo.COLLARDESIGN(CollarDesignID);
-- #8
INSERT INTO SocialMedia.dbo.GENDER(GenderID,Gender)
VALUES (-1,NULL);
ALTER TABLE SocialMedia.dbo.PRODUCT
ADD CONSTRAINT FK_SProduct_SGender FOREIGN KEY (GenderID) REFERENCES SocialMedia.dbo.GENDER(GenderID);


-- COLOR
ALTER TABLE SocialMedia.dbo.COLOR
ADD CONSTRAINT FK_SColor_SProduct FOREIGN KEY (ProductNo) REFERENCES SocialMedia.dbo.PRODUCT(ProductNo);
ALTER TABLE SocialMedia.dbo.COLOR
ADD CONSTRAINT FK_SColor_SColorRGB FOREIGN KEY (ColorID) REFERENCES SocialMedia.dbo.COLORRGB(ColorID);

-- PRODUCTHISTORY
ALTER TABLE SocialMedia.dbo.PRODUCTHISTORY
ADD CONSTRAINT FK_SColor_SProducthistory FOREIGN KEY (ProductNo) REFERENCES SocialMedia.dbo.PRODUCT(ProductNo);


-------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------

Update SocialMedia.dbo.PRODUCT set ProductCategoryID = NULL



CREATE TABLE	NRGSearch.dbo.tempmm (
                    ProductΝο INT NOT NULL,
	                ImageBlob VARBINARY(MAX))

insert into SocialMedia.dbo.tempmm
values(1,0xffa)



CREATE TABLE ImageTable
(
    Id int,
    Name varchar(50) ,
    Photo varbinary(max) 
)

INSERT INTO ImageTable (Id, Name, Photo) 
SELECT 1, 'test', BulkColumn 
FROM Openrowset( Bulk 'C:\test.jpg', Single_Blob) as image